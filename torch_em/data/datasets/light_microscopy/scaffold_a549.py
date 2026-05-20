"""The Scaffold-A549 dataset contains 3D confocal fluorescence microscopy images
of A549 human lung cancer cells grown in a scaffold matrix, with one fully annotated
volume for nucleus instance segmentation evaluation.

NOTE: The dataset contains 20 unlabeled training volumes and
1 labeled test volume (sf_a549_21), each of shape 64 x 512 x 512.
Also, the labeled test volume isn't the best of annotation quality.

The dataset is located at https://github.com/Kaiseem/Scaffold-A549.
This dataset is from the publication https://doi.org/10.1007/s12559-021-09944-4.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Tuple, Union

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://github.com/Kaiseem/Scaffold-A549/releases/download/v1.0/scaffold_a549.zip"
CHECKSUM = None


def get_scaffold_a549_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Scaffold-A549 dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data directory.
    """
    data_dir = os.path.join(path, "scaffold_a549")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "scaffold_a549.zip")
    util.download_source(zip_path, URL, download, checksum=CHECKSUM)
    util.unzip(zip_path, path)

    return data_dir


def _convert_to_tif(data_dir):
    """Convert .npy volumes to .tif for compatibility with torch_em loaders."""
    import imageio.v3 as imageio

    for subdir in ("train", "test"):
        npy_files = natsorted(glob(os.path.join(data_dir, subdir, "*.npy")))
        for npy_path in npy_files:
            tif_path = npy_path.replace(".npy", ".tif")
            if not os.path.exists(tif_path):
                arr = np.load(npy_path)
                imageio.imwrite(tif_path, arr)


def get_scaffold_a549_paths(
    path: Union[os.PathLike, str],
    split: str = "test",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Scaffold-A549 data.

    Note: Only the test split has ground truth labels. The train split contains
    unlabeled volumes only.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use. Either 'train' (unlabeled) or 'test' (labeled).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data (empty list for 'train' split).
    """
    if split not in ("train", "test"):
        raise ValueError(f"'{split}' is not a valid split. Choose 'train' or 'test'.")

    data_dir = get_scaffold_a549_data(path, download)
    _convert_to_tif(data_dir)

    split_dir = os.path.join(data_dir, split)
    if split == "test":
        raw_paths = [os.path.join(split_dir, "sf_a549_21.tif")]
        label_paths = [os.path.join(split_dir, "sf_a549_21_Label.tif")]
    else:
        raw_paths = natsorted(glob(os.path.join(split_dir, "sf_a549_*.tif")))
        raw_paths = [p for p in raw_paths if "Label" not in p]
        label_paths = []

    return raw_paths, label_paths


def get_scaffold_a549_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: str = "test",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Scaffold-A549 dataset for 3D nucleus instance segmentation.

    Note: Only the test split has ground truth labels. The train split contains
    20 unlabeled volumes that can be used for self-supervised learning.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The split to use. Either 'train' (unlabeled) or 'test' (labeled).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_scaffold_a549_paths(path, split, download)

    if split == "test":
        return torch_em.default_segmentation_dataset(
            raw_paths=raw_paths,
            raw_key=None,
            label_paths=label_paths,
            label_key=None,
            patch_shape=patch_shape,
            **kwargs,
        )
    else:
        return torch_em.default_segmentation_dataset(
            raw_paths=raw_paths,
            raw_key=None,
            label_paths=None,
            label_key=None,
            patch_shape=patch_shape,
            **kwargs,
        )


def get_scaffold_a549_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: str = "test",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the Scaffold-A549 dataloader for 3D nucleus instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The split to use. Either 'train' (unlabeled) or 'test' (labeled).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_scaffold_a549_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
