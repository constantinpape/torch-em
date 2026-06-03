"""This dataset contains crops of EM data with annotated nuclei from the adult fruit fly brain.

The data is extracted from https://doi.org/10.1101/2021.11.04.467197, which contains a segmentation
of all nuclei in the fruit fly brain. Please cite it if you use this dataset for a publication.
"""

import os
from glob import glob
from typing import Tuple, Union, Literal, List

import torch_em

from torch.utils.data import Dataset, DataLoader

from .. import util


URL = "https://owncloud.gwdg.de/index.php/s/PDrkO02w7FWLrsh/download"
CHECKSUM = "7f245866fa8ffb473f7bda1fbbea23d265183e4ada5e34a36ecd60475809dca3"


def get_fafb_nuclei_data(path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool) -> str:
    """Download the FAFB Nucleus data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use. One of 'train', 'val', 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the downloaded data.
    """
    assert split in ("train", "val", "test")
    split_folder = os.path.join(path, split)
    if not os.path.exists(split_folder):
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, "fafb_nucleus_data.zip")
        util.download_source(zip_path, URL, download, CHECKSUM)
        util.unzip(zip_path, path, remove=True)
    return split_folder


def get_fafb_nuclei_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool
) -> List[str]:
    """Get paths to the FAFB Nucleus data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use. One of 'train', 'val', 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the stored data.
    """
    get_fafb_nuclei_data(path, split, download)
    split_folder = os.path.join(path, split)
    paths = sorted(glob(os.path.join(split_folder, "*.h5")))
    return paths


def get_fafb_nuclei_dataset(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    patch_shape: Tuple[int, int, int],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the FAFB nucleus dataset for the segmentation of nuclei in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split for the dataset, either 'train, 'val', or 'test'.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    paths = get_fafb_nuclei_paths(path, split, download)
    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels/nuclei",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_fafb_nuclei_loader(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the FAFB nucleus dataloader for the segmentation of nuclei in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split for the dataset, either 'train', 'val', or 'test'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
       The segmentation dataset.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_fafb_nuclei_dataset(path, split, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
