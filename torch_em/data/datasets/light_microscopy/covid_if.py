"""This dataset contains annotation for cell and nucleus segmentation
in immunofluorescence microscopy.

This dataset is from the publication https://doi.org/10.1002/bies.202000257.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import List, Optional, Tuple, Union

import torch_em
from torch.utils.data import Dataset, DataLoader
from .. import util

COVID_IF_URL = "https://zenodo.org/record/5092850/files/covid-if-groundtruth.zip?download=1"
CHECKSUM = "d9cd6c85a19b802c771fb4ff928894b19a8fab0e0af269c49235fdac3f7a60e1"


def get_covid_if_data(path: Union[os.PathLike, str], download: bool) -> str:
    """Download the CovidIF training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the training data.
    """
    url = COVID_IF_URL
    checksum = CHECKSUM

    if os.path.exists(path):
        return path

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "covid-if.zip")
    util.download_source(zip_path, url, download, checksum)
    util.unzip(zip_path, path, True)

    return path


def get_covid_if_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    sample_range: Optional[Tuple[int, int]] = None,
    target: str = "cells",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs
) -> Dataset:
    """Get the CovidIF dataset for segmenting nuclei or cells in immunofluorescence microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        sample_range: Id range of samples to load from the training dataset.
        target: The segmentation task. Either 'cells' or 'nuclei'.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    available_targets = ("cells", "nuclei")
    # TODO also support infected_cells
    # available_targets = ("cells", "nuclei", "infected_cells")
    assert target in available_targets, f"{target} not found in {available_targets}"

    if target == "cells":
        raw_key = "raw/serum_IgG/s0"
        label_key = "labels/cells/s0"
    elif target == "nuclei":
        raw_key = "raw/nuclei/s0"
        label_key = "labels/nuclei/s0"

    get_covid_if_data(path, download)

    file_paths = sorted(glob(os.path.join(path, "*.h5")))
    if sample_range is not None:
        start, stop = sample_range
        if start is None:
            start = 0
        if stop is None:
            stop = len(file_paths)
        file_paths = [os.path.join(path, f"gt_image_{idx:03}.h5") for idx in range(start, stop)]
        assert all(os.path.exists(fp) for fp in file_paths), f"Invalid sample range {sample_range}"

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )
    kwargs = util.update_kwargs(kwargs, "ndim", 2)

    return torch_em.default_segmentation_dataset(
        file_paths, raw_key, file_paths, label_key, patch_shape, **kwargs
    )


def get_covid_if_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    sample_range: Optional[Tuple[int, int]] = None,
    target: str = "cells",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CovidIF dataloder for segmenting nuclei or cells in immunofluorescence microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        sample_range: Id range of samples to load from the training dataset.
        target: The segmentation task. Either 'cells' or 'nuclei'.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_covid_if_dataset(
        path, patch_shape, sample_range=sample_range, target=target, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary, **ds_kwargs,
    )
    loader = torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
    return loader
