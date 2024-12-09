"""The NuClick dataset contains annotations for lymphocytes in IHC histopathology images.

This dataset is located at https://warwick.ac.uk/fac/cross_fac/tia/data/nuclick/.
The dataset is from the publication http://www.sciencedirect.com/science/article/pii/S1361841520301353.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Tuple, List, Literal, Union

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://warwick.ac.uk/fac/cross_fac/tia/data/nuclick/ihc_nuclick.zip"
CHECKSUM = "5128f1dfcba531e89b49e26364bc667eeb9978fa0039baa25a7f73fdaec2d736"


def get_nuclick_data(path: Union[os.PathLike, str], download: bool = False):
    """Download the NuClick dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        donwload: Whether to download the data if it is not present.

    Returns:
        Filepath where the dataset is downloaded.
    """
    data_dir = os.path.join(path, "IHC_nuclick", "IHC")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "ihc_nuclick.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_nuclick_paths(
    path: Union[os.PathLike, str], split: Literal["Train", "Validation"], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the NuClick data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use for the dataset. Either 'Train' or 'Validation'.
        donwload: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_nuclick_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "images", split, "*.png")))
    label_paths = natsorted(glob(os.path.join(data_dir, "masks", split, "*.npy")))

    neu_label_paths = []
    for lpath in tqdm(label_paths):
        neu_lpath = Path(lpath).with_suffix(".tif")
        neu_label_paths.append(str(neu_lpath))
        if os.path.exists(neu_lpath):
            continue

        imageio.imwrite(neu_lpath, np.load(lpath), compression="zlib")

    return raw_paths, neu_label_paths


def get_nuclick_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["Train", "Validation"],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the NuClick dataset for lymphocyte segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The split to use for the dataset. Either 'Train' or 'Validation'.
        donwload: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_nuclick_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        with_channels=True,
        ndim=2,
        patch_shape=patch_shape,
        **kwargs
    )


def get_nuclick_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["Train", "Validation"],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the NuClick dataloader for lymphocyte segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The split to use for the dataset. Either 'Train' or 'Validation'.
        donwload: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_nuclick_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
