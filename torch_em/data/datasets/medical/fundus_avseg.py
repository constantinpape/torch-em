"""The Fundus AVSeg dataset contains annotations for artery-vein segmentation in
fundus images.

This dataset is from the publication https://doi.org/10.1038/s41597-025-05381-2.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import pandas as pd

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://figshare.com/ndownloader/files/54093641"
CHECKSUM = "6db5ff43c4e9c25aa93093aa295c67b10fa0c089ac650df6665c7a6bbae9539f"


def get_fundus_avseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Fundus AVSeg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Folder where the data is stored.
    """
    data_dir = os.path.join(path, "Fundus-AVSeg")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "Fundus_AVSeg.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_fundus_avseg_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Download the Fundus AVSeg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_fundus_avseg_data(path, download)

    breakpoint()

    raw_paths = ...
    label_paths = ...

    return raw_paths, label_paths


def get_fundus_avseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Fundus AVSeg dataset for artery-vein segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_fundus_avseg_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        ndim=2,
        **kwargs
    )


def get_fundus_avseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Fundus AVSeg dataloader for artery-vein segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_fundus_avseg_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
