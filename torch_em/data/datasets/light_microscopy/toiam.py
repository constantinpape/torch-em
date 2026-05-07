"""The TOIAM dataset contains annotations for microbial cell segmentation in
phase contrast microscopy images of microbial live-cell images (MLCI).

The dataset is located at https://doi.org/10.5281/zenodo.7260137.
This dataset is from the publicaiton https://arxiv.org/html/2411.00552v1.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/7260137/files/ctc_format.zip"
CHECKSUM = "9ec73277b29f2b032037d9e07c73c428ff51456c23a5866bf214bf5a71590c31"


def get_toiam_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the TOIAM dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "ctc_format.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def get_toiam_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the TOIAM data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_toiam_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "0*", "*.tif")))
    label_paths = natsorted(glob(os.path.join(data_dir, "0*_GT", "SEG", "man_*.tif")))

    return raw_paths, label_paths


def get_toiam_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, int], download: bool = False, **kwargs
) -> Dataset:
    """Get the TOIAM dataset for microbial cell segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_toiam_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_toiam_loader(
    path: Union[os.PathLike, str], batch_size: int, patch_shape: Tuple[int, int], download: bool = False, **kwargs
) -> DataLoader:
    """Get the TOIAM dataloader for microbial cell segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_toiam_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
