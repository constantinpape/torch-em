"""
"""

import os
import shutil
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/11456029/files/NIS3D.zip"
CHECKSUM = "3eb60b48eba87a5eeb71e9676d6df64296adc3dd93234a1db80cd9a0da28cd83"


def get_nis3d_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the NIS3D dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath for the downloaded data.
    """
    data_dir = os.path.join(path, "NIS3D")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "NIS3D.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    util.unzip(zip_path, path)

    # NOTE: For "MusMusculus_2", the ground truth labels are named oddly. We need to fix it manually.
    gt_path = os.path.join(data_dir, "NIS3D", "MusMusculus_2", "gt.tif")
    shutil.move(src=gt_path, dst=gt_path.replace("gt", "GroundTruth"))

    return data_dir


def get_nis3d_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the NIS3D data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_nis3d_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "NIS3D", "*", "data.tif")))
    label_paths = natsorted(glob(os.path.join(data_dir, "NIS3D", "*", "GroundTruth.tif")))

    assert len(raw_paths) and len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def get_nis3d_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, ...], download: bool = False, **kwargs,
) -> Dataset:
    """Get the NIS3D dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """

    raw_paths, label_paths = get_nis3d_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=True,
        patch_shape=patch_shape,
        **kwargs
    )


def get_nis3d_loader(
    path: Union[os.PathLike, str], batch_size: int, patch_shape: Tuple[int, ...], download: bool = False, **kwargs,
) -> DataLoader:
    """Get the NIS3D dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_nis3d_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
