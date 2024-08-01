"""Dataset for the segmentation of different structures in EM volume of a
platynereis larve. Contains annotations for the segmentation of:
- Cuticle
- Cilia
- Cells
- Nuclei

This dataset is from the publication https://doi.org/10.1016/j.cell.2021.07.017.
Please cite it if you use this dataset for a publication.
"""

import os
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch_em
from torch.utils.data import Dataset, DataLoader
from .. import util

URLS = {
    "cells": "https://zenodo.org/record/3675220/files/membrane.zip",
    "nuclei": "https://zenodo.org/record/3675220/files/nuclei.zip",
    "cilia": "https://zenodo.org/record/3675220/files/cilia.zip",
    "cuticle": "https://zenodo.org/record/3675220/files/cuticle.zip"
}

CHECKSUMS = {
    "cells": "30eb50c39e7e9883e1cd96e0df689fac37a56abb11e8ed088907c94a5980d6a3",
    "nuclei": "a05033c5fbc6a3069479ac6595b0a430070f83f5281f5b5c8913125743cf5510",
    "cilia": "6d2b47f63d39a671789c02d8b66cad5e4cf30eb14cdb073da1a52b7defcc5e24",
    "cuticle": "464f75d30133e8864958049647fe3c2216ddf2d4327569738ad72d299c991843"
}


#
# TODO data-loader for more classes:
# - mitos
#


def _check_data(path, prefix, extension, n_files):
    if not os.path.exists(path):
        return False
    files = glob(os.path.join(path, f"{prefix}*{extension}"))
    return len(files) == n_files


def _get_paths_and_rois(sample_ids, n_files, template, rois):
    if sample_ids is None:
        sample_ids = list(range(1, n_files + 1))
    else:
        assert min(sample_ids) >= 1 and max(sample_ids) <= n_files
        sample_ids.sort()
    paths = [template % sample for sample in sample_ids]
    data_rois = [rois.get(sample, np.s_[:, :, :]) for sample in sample_ids]
    return paths, data_rois


def get_platy_data(path: Union[os.PathLike, str], name: str, download: bool) -> Tuple[str, int]:
    """Download the platynereis dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: Name of the segmentation task. Available tasks: 'cuticle', 'cilia', 'cells' or 'nuclei'.
        download: Whether to download the data if it is not present.

    Returns:
        The path to the folder where the data has been downloaded.
        The number of files downloaded.
    """
    data_root = os.path.join(path, name)

    if name == "cuticle":
        ext, prefix, n_files = ".n5", "train_data_", 5
    elif name == "cilia":
        ext, prefix, n_files = ".h5", "train_data_cilia_", 3
    elif name == "cells":
        data_root = os.path.join(path, "membrane")
        ext, prefix, n_files = ".n5", "train_data_membrane_", 9
    elif name == "nuclei":
        ext, prefix, n_files = ".h5", "train_data_nuclei_", 12
    else:
        raise ValueError(f"Invalid name {name}. Expect one of 'cuticle', 'cilia', 'cell' or 'nuclei'.")

    data_is_complete = _check_data(data_root, prefix, ext, n_files)
    if data_is_complete:
        return data_root, n_files

    os.makedirs(path, exist_ok=True)
    url = URLS[name]
    checksum = CHECKSUMS[name]

    zip_path = os.path.join(path, f"data-{name}.zip")
    util.download_source(zip_path, url, download=download, checksum=checksum)
    util.unzip(zip_path, path, remove=True)

    return data_root, n_files


def get_platynereis_cuticle_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    sample_ids: Optional[Sequence[int]] = None,
    download: bool = False,
    rois: Dict[int, Any] = {},
    **kwargs
) -> Dataset:
    """Get the dataset for cuticle segmentation in platynereis.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        sample_ids: The sample ids to use for the dataset
        download: Whether to download the data if it is not present.
        rois: The region of interests to use for the data blocks.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    cuticle_root, n_files = get_platy_data(path, "cuticle", download)

    paths, data_rois = _get_paths_and_rois(sample_ids, n_files, os.path.join(cuticle_root, "train_data_%02i.n5"), rois)
    raw_key, label_key = "volumes/raw", "volumes/labels/segmentation"
    return torch_em.default_segmentation_dataset(
        paths, raw_key, paths, label_key, patch_shape, rois=data_rois, **kwargs
    )


def get_platynereis_cuticle_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    sample_ids: Optional[Sequence[int]] = None,
    download: bool = False,
    rois: Dict[int, Any] = {},
    **kwargs
) -> DataLoader:
    """Get the dataloader for cuticle segmentation in platynereis.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        sample_ids: The sample ids to use for the dataset
        download: Whether to download the data if it is not present.
        rois: The region of interests to use for the data blocks.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_platynereis_cuticle_dataset(
        path, patch_shape, sample_ids=sample_ids, download=download, rois=rois, **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)


def get_platynereis_cilia_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    sample_ids: Optional[Sequence[int]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    rois: Dict[int, Any] = {},
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the dataset for cilia segmentation in platynereis.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        sample_ids: The sample ids to use for the dataset
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        rois: The region of interests to use for the data blocks.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    cilia_root, n_files = get_platy_data(path, "cilia", download)

    paths, rois = _get_paths_and_rois(sample_ids, n_files, os.path.join(cilia_root, "train_data_cilia_%02i.h5"), rois)
    raw_key = "volumes/raw"
    label_key = "volumes/labels/segmentation"

    kwargs = util.update_kwargs(kwargs, "rois", rois)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, boundaries=boundaries, offsets=offsets, binary=binary,
    )
    return torch_em.default_segmentation_dataset(paths, raw_key, paths, label_key, patch_shape, **kwargs)


def get_platynereis_cilia_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    sample_ids: Optional[Sequence[int]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    rois: Dict[int, Any] = {},
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the dataloader for cilia segmentation in platynereis.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        sample_ids: The sample ids to use for the dataset
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        rois: The region of interests to use for the data blocks.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_platynereis_cilia_dataset(
        path, patch_shape, sample_ids=sample_ids,
        offsets=offsets, boundaries=boundaries, binary=binary,
        rois=rois, download=download, **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)


def get_platynereis_cell_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    sample_ids: Optional[Sequence[int]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    rois: Dict[int, Any] = {},
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the dataset for cell segmentation in platynereis.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        sample_ids: The sample ids to use for the dataset
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        rois: The region of interests to use for the data blocks.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    cell_root, n_files = get_platy_data(path, "cells", download)

    template = os.path.join(cell_root, "train_data_membrane_%02i.n5")
    data_paths, data_rois = _get_paths_and_rois(sample_ids, n_files, template, rois)

    kwargs = util.update_kwargs(kwargs, "rois", data_rois)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets,
    )

    raw_key = "volumes/raw/s1"
    label_key = "volumes/labels/segmentation/s1"
    return torch_em.default_segmentation_dataset(data_paths, raw_key, data_paths, label_key, patch_shape,  **kwargs)


def get_platynereis_cell_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    sample_ids: Optional[Sequence[int]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    rois: Dict[int, Any] = {},
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the dataloader for cell segmentation in platynereis.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        sample_ids: The sample ids to use for the dataset
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        rois: The region of interests to use for the data blocks.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_platynereis_cell_dataset(
        path, patch_shape, sample_ids, rois=rois,
        offsets=offsets, boundaries=boundaries, download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)


def get_platynereis_nuclei_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    sample_ids: Optional[Sequence[int]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    rois: Dict[int, Any] = {},
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the dataset for nucleus segmentation in platynereis.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        sample_ids: The sample ids to use for the dataset
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        rois: The region of interests to use for the data blocks.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    nuc_root, n_files = get_platy_data(path, "nuclei", download)

    if sample_ids is None:
        sample_ids = list(range(1, n_files + 1))
    assert min(sample_ids) >= 1 and max(sample_ids) <= n_files
    sample_ids.sort()

    template = os.path.join(nuc_root, "train_data_nuclei_%02i.h5")
    data_paths, data_rois = _get_paths_and_rois(sample_ids, n_files, template, rois)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs = util.update_kwargs(kwargs, "rois", data_rois)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, boundaries=boundaries, offsets=offsets, binary=binary,
    )

    raw_key = "volumes/raw"
    label_key = "volumes/labels/nucleus_instance_labels"
    return torch_em.default_segmentation_dataset(data_paths, raw_key, data_paths, label_key, patch_shape, **kwargs)


def get_platynereis_nuclei_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    sample_ids: Optional[Sequence[int]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    rois: Dict[int, Any] = {},
    download: bool = False,
    **kwargs
):
    """Get the dataloader for nucleus segmentation in platynereis.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        sample_ids: The sample ids to use for the dataset
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        rois: The region of interests to use for the data blocks.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_platynereis_nuclei_dataset(
        path, patch_shape, sample_ids=sample_ids, rois=rois,
        offsets=offsets, boundaries=boundaries, binary=binary, download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
