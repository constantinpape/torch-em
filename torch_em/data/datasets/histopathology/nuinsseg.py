"""The NuInsSeg dataset contains annotations for nucleus segmentation in
H&E stained histology images for 31 organs across humans and rats.

The dataset is located at https://www.kaggle.com/datasets/ipateam/nuinsseg.
This dataset is from the publication https://doi.org/10.1038/s41597-024-03117-2.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Tuple, Union, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def get_nuinsseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the NuInsSeg dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name="ipateam/nuinsseg", download=download)
    util.unzip(zip_path=os.path.join(path, "nuinsseg.zip"), dst=data_dir)

    return data_dir


def get_nuinsseg_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the NuInsSeg data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_nuinsseg_data(path, download)

    tissue_type_dirs = glob(os.path.join(data_dir, "*"))
    raw_paths = [p for dir in tissue_type_dirs for p in natsorted(glob(os.path.join(dir, "tissue images", "*.png")))]
    label_paths = [
        p for dir in tissue_type_dirs for p in natsorted(glob(os.path.join(dir, "label masks modify", "*.tif")))
    ]

    return raw_paths, label_paths


def get_nuinsseg_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, int], download: bool = False, **kwargs
) -> Dataset:
    """Get the NuInsSeg dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_nuinsseg_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs
    )


def get_nuinsseg_loader(
    path: Union[os.PathLike, str], batch_size: int, patch_shape: Tuple[int, int], download: bool = False, **kwargs
) -> DataLoader:
    """Get the NuInsSeg dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_nuinsseg_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
