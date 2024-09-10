"""
"""

import os
from typing import List, Optional, Tuple, Union

import torch_em
from .. import util


ACTIN_ID = 10002


def get_deepict_actin_data(path: Union[os.PathLike, str], download: bool) -> str:
    """Download the deepict actin dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The path to the downloaded data.
    """
    # TODO check if the processed data is there already

    util.download_from_cryo_et_portal(path, ACTIN_ID, download)
    # TODO process the data

    return path


def get_deepict_actin_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    download: bool = False,
    **kwargs
):
    """Get the dataset for EM neuron segmentation in ISBI 2012.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    assert len(patch_shape) == 3
    data_path = get_deepict_actin_data(path, download)

    raw_key = "raw"
    label_key = "labels/membranes"

    return torch_em.default_segmentation_dataset(data_path, raw_key, data_path, label_key, patch_shape, **kwargs)


def get_deepict_actin_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    download: bool = False,
    **kwargs
):
    """Get the DataLoader for EM neuron segmentation in ISBI 2012.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_deepict_actin_loader(
        path, patch_shape, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
