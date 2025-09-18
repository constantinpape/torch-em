"""The BacMother dataset contains bacteria annotations for E. Coli videos.
This dataset also has tracking annotations in CTC format.

The dataset is hosted on Zenodo https://doi.org/10.5281/zenodo.11237127.
The dataset is from the publication https://doi.org/10.1371/journal.pcbi.1013071.

Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Union, Tuple, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/11237127/files/CTC.zip"
CHECKSUM = "280f4cacda12094b6eafaae772ce7ea25f8ad6093d2ec2b3d381504dbea70ed3"


def get_bac_mother_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BacMother dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the dataset is stored.
    """
    data_dir = os.path.join(path, "CTC")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "CTC.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_bac_mother_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths for the BacMother dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_path = get_bac_mother_data(path, download)

    assert split in ["train", "val", "test"], f"'{split}' is not a valid data split."
    data_path = os.path.join(data_path, split)

    raw_dirs = [p for p in glob(os.path.join(data_path, "*")) if not p.endswith("_GT")]

    raw_paths, label_paths = [], []
    for raw_dir in raw_dirs:
        raw_paths.extend(natsorted(glob(os.path.join(raw_dir, "t*.tif"))))
        label_paths.extend(natsorted(glob(os.path.join(f"{raw_dir}_GT", "SEG", "man_seg*.tif"))))

    assert raw_paths and len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def get_bac_mother_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BacMother dataset for segmentation of bacteria.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_bac_mother_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        ndim=2,
        with_channels=True,
        is_seg_dataset=False,
        patch_shape=patch_shape,
    )


def get_bac_mother_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BacMother dataloader for segmentation of bacteria.

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
    dataset = get_bac_mother_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
