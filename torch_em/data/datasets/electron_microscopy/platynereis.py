"""Dataset for the segmentation of different structures in EM volume of a
platynereis larve. Contains annotations for the segmentation of:
- Cuticle
- Cilia
- Cells
- Nuclei

This dataset is from the publication https://doi.org/10.1016/j.cell.2021.07.017.
Please cite it if you use this dataset for a publication.

The cell dataset stores corrected labels separately from the source annotations. It maps the
neuropil IDs in `CELL_NEUROPIL_IDS` to `ignore_label` before sampling training patches.
"""

import os
from glob import glob
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from elf.io import open_file
from skimage.segmentation import flood

from torch.utils.data import Dataset, DataLoader

import torch_em
from torch_em.data import ConcatDataset

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

FILE_TEMPLATES = {
    "cells": "train_data_membrane_%02i.n5",
    "nuclei": "train_data_nuclei_%02i.h5",
    "cilia": "train_data_cilia_%02i.h5",
    "cuticle": "train_data_%02i.n5",
}

# Default ignore label. Stays exact through a float32 round-trip, which the augmentations do,
# and sits far above any real instance id.
CELL_IGNORE_LABEL = 2 ** 24 - 1
# Increment this version when corrections change, so stored labels and ROI caches are regenerated.
CELL_LABEL_VERSION = 1
CELL_SOURCE_LABEL_KEY = "volumes/labels/segmentation/s1"
CELL_BACKGROUND_IDS = {7: (63,), 8: (5,)}

# Ids that label neuropil rather than a single cell, found by inspecting every cell volume.
CELL_NEUROPIL_IDS = {
    1: (),
    2: (),
    3: (253,),
    4: (58,),
    5: (72,),
    6: (73,),
    7: (),
    8: (43,),
    9: (19, 1084),
}


def get_platynereis_cell_neuropil_ids(sample_id: int) -> Tuple[int, ...]:
    """Get the neuropil instance ids of a platynereis cell volume.

    Args:
        sample_id: The id of the volume, between 1 and 9.

    Returns:
        The instance ids that label neuropil rather than a single cell. Empty if the volume has none.
    """
    return CELL_NEUROPIL_IDS.get(sample_id, ())


def get_platynereis_cell_label_key(ignore_label: int = CELL_IGNORE_LABEL) -> str:
    """Get the N5 key for corrected cell labels.

    Args:
        ignore_label: The value assigned to neuropil voxels.

    Returns:
        The dataset key, including the correction version and ignore label.
    """
    return f"volumes/labels/segmentation_corrected/v{CELL_LABEL_VERSION}/ignore_{ignore_label}/s1"


def _split_cell_label(labels, source_id, target_id, seed, expected_size, expected_bbox):
    if any(c >= size for c, size in zip(seed, labels.shape)) or labels[seed] != source_id:
        raise ValueError("The muscle seed does not point to the expected source label.")
    if np.any(labels == target_id):
        raise ValueError(f"The muscle target label {target_id} is already assigned.")
    muscle = flood(labels, seed, connectivity=3)
    coordinates = np.where(muscle)
    bbox = tuple((int(c.min()), int(c.max()) + 1) for c in coordinates)
    if int(muscle.sum()) != expected_size or bbox != expected_bbox:
        raise ValueError("The muscle component differs from the inspected size or bounding box.")
    labels[muscle] = target_id


def _correct_cell_labels(labels, sample_id, ignore_label):
    labels = labels.astype("int64" if ignore_label < 0 else "uint64", copy=True)
    if ignore_label != 0 and np.any(labels == ignore_label):
        raise ValueError(f"The ignore label {ignore_label} is already assigned to an instance.")
    if sample_id == 5:
        if ignore_label == 256:
            raise ValueError("The ignore label conflicts with the new muscle label 256.")
        _split_cell_label(
            labels, source_id=72, target_id=256, seed=(29, 70, 242), expected_size=102740,
            expected_bbox=((15, 80), (64, 129), (185, 293)),
        )
    for label_id in CELL_BACKGROUND_IDS.get(sample_id, ()):
        labels[labels == label_id] = 0
    for label_id in get_platynereis_cell_neuropil_ids(sample_id):
        labels[labels == label_id] = ignore_label
    return labels


def _prepare_cell_labels(path, sample_id, ignore_label):
    key = get_platynereis_cell_label_key(ignore_label)
    target = os.path.join(path, key)
    if os.path.exists(target):
        return
    with open_file(path, "r") as f:
        source = f[CELL_SOURCE_LABEL_KEY]
        labels = _correct_cell_labels(source[:], sample_id, ignore_label)
        chunks = source.chunks
        spatial_attrs = {name: source.attrs[name] for name in ("offset", "global_offset") if name in source.attrs}

    # Publish the complete dataset atomically. Other training ranks may prepare it at the same time.
    with TemporaryDirectory(dir=path, prefix=".cell-labels-") as tmp:
        tmp_path = os.path.join(tmp, "labels.n5")
        with open_file(tmp_path, "a") as f:
            ds = f.create_dataset("labels", data=labels, chunks=chunks, compression="gzip")
            ds.attrs.update(spatial_attrs)
            ds.attrs.update({
                "correction_version": CELL_LABEL_VERSION,
                "source_key": CELL_SOURCE_LABEL_KEY,
                "sample_id": sample_id,
                "ignore_label": ignore_label,
                "neuropil_ids": list(get_platynereis_cell_neuropil_ids(sample_id)),
                "background_ids": list(CELL_BACKGROUND_IDS.get(sample_id, ())),
            })
        os.makedirs(os.path.dirname(target), exist_ok=True)
        try:
            os.rename(os.path.join(tmp_path, "labels"), target)
        except OSError:
            if not os.path.isdir(target):
                raise


def prepare_platynereis_cell_data(
    path: Union[os.PathLike, str],
    sample_ids: Optional[Sequence[int]] = None,
    download: bool = False,
    ignore_label: int = CELL_IGNORE_LABEL,
) -> List[str]:
    """Prepare corrected cell labels without changing the source annotations.

    Corrections split the muscle cell in volume 5, remove false foreground in volumes 7 and 8,
    and map neuropil IDs to the ignore label. Existing labels for this version are reused.

    Args:
        path: Folder containing the membrane subfolder.
        sample_ids: Volume IDs to prepare. By default, prepare all nine volumes.
        download: Whether to download missing source data.
        ignore_label: The value assigned to neuropil voxels.

    Returns:
        The N5 paths in sample ID order. Read corrected labels with `get_platynereis_cell_label_key`.
    """
    sample_ids = list(range(1, 10)) if sample_ids is None else sorted(sample_ids)
    paths = get_platynereis_paths(path, sample_ids, name="cells", download=download)
    for sample_id, data_path in zip(sample_ids, paths):
        _prepare_cell_labels(data_path, sample_id, ignore_label)
    return paths


#
# TODO data-loader for more classes:
# - mitos
#


def _check_data(path, prefix, extension, n_files):
    if not os.path.exists(path):
        return False
    files = glob(os.path.join(path, f"{prefix}*{extension}"))
    return len(files) == n_files


def get_platynereis_data(path: Union[os.PathLike, str], name: str, download: bool) -> Tuple[str, int]:
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


def get_platynereis_paths(path, sample_ids, name, rois={}, download=False, return_rois=False):
    """Get paths to the platynereis data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        sample_ids: The sample ids to use for the dataset
        name: Name of the segmentation task. Available tasks: 'cuticle', 'cilia', 'cells' or 'nuclei'.
        rois: The region of interests to use for the data blocks.
        download: Whether to download the data if it is not present.
        return_rois: Whether to return the extracted rois.

    Returns:
        The filepaths for the stored data.
    """
    root, n_files = get_platynereis_data(path, name, download)
    template = os.path.join(root, FILE_TEMPLATES[name])

    if sample_ids is None:
        sample_ids = list(range(1, n_files + 1))
    else:
        assert min(sample_ids) >= 1 and max(sample_ids) <= n_files
        sample_ids.sort()
    paths = [template % sample for sample in sample_ids]
    data_rois = [rois.get(sample, np.s_[:, :, :]) for sample in sample_ids]

    if return_rois:
        return paths, data_rois
    else:
        return paths


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
    paths, data_rois = get_platynereis_paths(
        path=path, sample_ids=sample_ids, name="cuticle", rois=rois, download=download, return_rois=True,
    )
    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="volumes/raw",
        label_paths=paths,
        label_key="volumes/labels/segmentation",
        patch_shape=patch_shape,
        rois=data_rois,
        **kwargs
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
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
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
    paths, rois = get_platynereis_paths(
        path=path, sample_ids=sample_ids, name="cilia", rois=rois, download=download, return_rois=True,
    )
    kwargs = util.update_kwargs(kwargs, "rois", rois)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, boundaries=boundaries, offsets=offsets, binary=binary,
    )
    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="volumes/raw",
        label_paths=paths,
        label_key="volumes/labels/segmentation",
        patch_shape=patch_shape,
        **kwargs
    )


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
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
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
    ignore_label: int = CELL_IGNORE_LABEL,
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
        ignore_label: The value the neuropil ids of `CELL_NEUROPIL_IDS` are mapped to, so that a loss
            can exclude them.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    data_paths, data_rois = get_platynereis_paths(
        path=path, sample_ids=sample_ids, name="cells", rois=rois, download=download, return_rois=True,
    )
    prepare_platynereis_cell_data(path, sample_ids, download=download, ignore_label=ignore_label)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets,
    )

    ds_kwargs = dict(
        raw_key="volumes/raw/s1", label_key=get_platynereis_cell_label_key(ignore_label), patch_shape=patch_shape,
    )

    datasets = []
    for data_path, data_roi in zip(data_paths, data_rois):
        datasets.append(
            torch_em.default_segmentation_dataset(
                raw_paths=[data_path], label_paths=[data_path], rois=[data_roi], **ds_kwargs, **kwargs
            )
        )

    return datasets[0] if len(datasets) == 1 else ConcatDataset(*datasets)


def get_platynereis_cell_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    sample_ids: Optional[Sequence[int]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    rois: Dict[int, Any] = {},
    download: bool = False,
    ignore_label: int = CELL_IGNORE_LABEL,
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
        ignore_label: The value the neuropil ids of `CELL_NEUROPIL_IDS` are mapped to.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_platynereis_cell_dataset(
        path, patch_shape, sample_ids, rois=rois,
        offsets=offsets, boundaries=boundaries, download=download, ignore_label=ignore_label,
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
    _, n_files = get_platynereis_data(path, "nuclei", download)

    if sample_ids is None:
        sample_ids = list(range(1, n_files + 1))
    assert min(sample_ids) >= 1 and max(sample_ids) <= n_files
    sample_ids.sort()

    data_paths, data_rois = get_platynereis_paths(
        path=path, sample_ids=sample_ids, name="nuclei", rois=rois, download=download, return_rois=True,
    )

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs = util.update_kwargs(kwargs, "rois", data_rois)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, boundaries=boundaries, offsets=offsets, binary=binary,
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="volumes/raw",
        label_paths=data_paths,
        label_key="volumes/labels/nucleus_instance_labels",
        patch_shape=patch_shape,
        **kwargs
    )


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
) -> DataLoader:
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
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_platynereis_nuclei_dataset(
        path, patch_shape, sample_ids=sample_ids, rois=rois,
        offsets=offsets, boundaries=boundaries, binary=binary, download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
