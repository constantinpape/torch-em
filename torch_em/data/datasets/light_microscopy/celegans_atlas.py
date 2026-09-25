"""CElegans Atlas is a dataset that contains nucleus annotations in 3d confocal microscopy images.

The preprocessed dataset is located at https://zenodo.org/records/5942575.
The raw images are from the publication https://doi.org/10.1038/nmeth.1366.
The nucleus annotation masks were generated in the publication https://arxiv.org/abs/2002.02857.
And the available data splits were made by the following publication https://arxiv.org/abs/1908.03636.

Please cite them all if you use this dataset for your research.
"""

import os
import shutil
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/5942575/files/c_elegans_nuclei.zip"
CHECKSUM = "1def07491cdad89e381cbe4437ef03da3af8f78d127e8152cd9b32bdab152c4e"


def get_celegans_atlas_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CElegans Atlas dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the dataset is stored.
    """
    data_dir = os.path.join(path, "c_elegans_nuclei")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    # Download and unzip the images.
    zip_path = os.path.join(path, "c_elegans_nuclei.zip")
    util.download_source(zip_path, url=URL, checksum=CHECKSUM, download=download)
    util.unzip(zip_path, path)

    # Remove other miscellanous folders.
    shutil.rmtree(os.path.join(path, "__MACOSX"))

    return data_dir


def get_celegans_atlas_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CElegans Atlas data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in ["train", "val", "test"]:
        raise ValueError(f"'{split}' is not a valid data split choice.")

    data_path = get_celegans_atlas_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_path, split, "images", "*.tif")))
    label_paths = natsorted(glob(os.path.join(data_path, split, "masks", "*.tif")))

    return raw_paths, label_paths


def get_celegans_atlas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the CElegans Atlas dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_celegans_atlas_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_celegans_atlas_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the CElegans Atlas dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_celegans_atlas_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
