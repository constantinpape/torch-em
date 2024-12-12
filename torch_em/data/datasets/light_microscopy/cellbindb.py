"""CellBinDB contains annotations for cell segmentation in multi-modal images.
- Consists of DAPI, ssDNA, H&E, and mIF staining.
- Covers more than 30 normal and diseased tissue types from human and mouse samples.

The dataset is located at https://db.cngb.org/search/project/CNP0006370/.
This dataset is from the publication https://doi.org/10.1101/2024.11.20.619750.
Please cite it if you use this dataset for your research.
"""

import os
import subprocess
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Optional

import torch_em

from torch.utils.data import Dataset, DataLoader

from .. import util
from .neurips_cell_seg import to_rgb


DOWNLOAD_SCRIPT = 'wget -c -nH -np -r -R "index.html*" --cut-dirs 4 ftp://ftp.cngb.org/pub/CNSA/data5/CNP0006370/Other/'

CHOICES = ["10×Genomics_DAPI", "10×Genomics_HE", "DAPI", "HE", "mIF", "ssDNA"]


def get_cellbindb_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CellBinDB dataset.

    Args:
        path: Filepath to a folder where the data is downloaded.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the data.
    """
    data_dir = os.path.join(path, "Other")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    if not download:
        raise AssertionError("The dataset is not found and download is set to 'False'.")

    print(
        "Downloading the dataset takes several hours and is extremely (like very very) slow. "
        "Make sure you have consistent internet connection or run it in background over a cluster."
    )
    splits = DOWNLOAD_SCRIPT.split(" ")
    subprocess.run([*splits[:-1], "-P", os.path.abspath(path), splits[-1]])
    return data_dir


def get_cellbindb_paths(
    path: Union[os.PathLike, str], data_choice: Optional[str] = None, download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the CellBinDB data.

    Args:
        path: Filepath to a folder where the data is downloaded.
        data_choice: The choice of dataset.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_cellbindb_data(path, download)

    if data_choice is None:
        data_choice = CHOICES
    else:
        assert data_choice in CHOICES
        if isinstance(data_choice, str):
            data_choice = [data_choice]

    raw_paths, label_paths = [], []
    for dchoice in data_choice:
        raw_paths.extend(natsorted(glob(os.path.join(data_dir, dchoice, "*", "*-img.tif"))))
        label_paths.extend(natsorted(glob(os.path.join(data_dir, dchoice, "*", "*-instancemask.tif"))))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_cellbindb_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    data_choice: Optional[str] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CellBinDB dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded.
        patch_shape: The patch shape to use for training.
        data_choice: The choice of dataset.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cellbindb_paths(path, data_choice, download)

    if "raw_transform" not in kwargs:
        kwargs["raw_transform"] = torch_em.transform.get_raw_transform(augmentation2=to_rgb)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        ndim=2,
        patch_shape=patch_shape,
        **kwargs
    )


def get_cellbindb_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    data_choice: Optional[str] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CellBinDB dataloader for cell segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded.
        patch_shape: The patch shape to use for training.
        data_choice: The choice of dataset.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cellbindb_dataset(path, patch_shape, data_choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
