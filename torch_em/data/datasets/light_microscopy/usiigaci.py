"""The Usiigaci dataset contains annotations for cell segmentation in
phase contrast microscopy images of NIH/3T3 fibroblasts.

This dataset is from the publication https://doi.org/10.1016/j.softx.2019.02.007.
Please cite it if you use this dataset for your research.
"""

import os
import subprocess
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def get_usiigaci_data(path: Union[os.PathLike, str], download: bool = False):
    """Download the Usiigaci dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.
    """
    data_dir = os.path.join(path, "Usiigaci")
    if os.path.exists(data_dir):
        return

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False")

    subprocess.run(["git", "clone", "--quiet", "https://github.com/oist/Usiigaci", data_dir])


def get_usiigaci_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'val'], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Usiigaci data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_usiigaci_data(path, download)

    # Labeled images.
    base_dir = os.path.join(path, "Usiigaci", r"Mask R-CNN", split, "set*")
    raw_paths = natsorted(glob(os.path.join(base_dir, "raw.tif")))
    label_paths = natsorted(glob(os.path.join(base_dir, "instances_ids.png")))

    if split == "train":
        # Example tracking data.
        base_dir = os.path.join(path, "Usiigaci", "ExampleData")
        raw_paths.extend(natsorted(glob(os.path.join(base_dir, "T98G_sample", "*.tif"))))
        label_paths.extend(natsorted(glob(os.path.join(base_dir, "T98G_sample_mask_avg", "*.png"))))

    assert len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def get_usiigaci_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val'],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Usiigaci dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_usiigaci_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs
    )


def get_usiigaci_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val'],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Usiigaci dataloader for cell segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_usiigaci_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
