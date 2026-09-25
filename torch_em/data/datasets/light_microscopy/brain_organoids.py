"""The Brain Organoids dataset contains annotations for organoid segmentation in
2d brightfield images of brain organoids.

This dataset is from the publication https://doi.org/10.1038/s41597-024-03330-z.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util
from .neurips_cell_seg import to_rgb


URL = "https://zenodo.org/records/10301912/files/data.zip"
CHECKSUM = "bc2ed56717a65ccd49e27bac92c3b714ca4bb245299698b68baa599c9d510a26"


def get_brain_organoids_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Brain Organoids dataset.

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

    zip_path = os.path.join(path, "data.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_brain_organoids_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get the Brain Organoids data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_brain_organoids_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "imgs", "*")))
    label_paths = natsorted(glob(os.path.join(data_dir, "labels", "*.npy")))

    preprocessed_dir = os.path.join(data_dir, "labels_preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    neu_label_paths = []
    for lpath in tqdm(label_paths, desc="Preprocessing labels"):
        neu_lpath = lpath.replace("labels", "labels_preprocessed").replace(".npy", ".tif")
        neu_label_paths.append(neu_lpath)
        if os.path.exists(neu_lpath):
            continue

        label = np.load(lpath)
        imageio.imwrite(neu_lpath, label)

    return raw_paths, neu_label_paths


def get_brain_organoids_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, int], download: bool = False, **kwargs
) -> Dataset:
    """Get the Brain Organoids dataset for organoid segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_brain_organoids_paths(path, download)

    if "raw_transform" not in kwargs:
        kwargs["raw_transform"] = torch_em.transform.get_raw_transform(augmentation2=to_rgb)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        with_channels=True,
        patch_shape=patch_shape,
        **kwargs
    )


def get_brain_organoids_loader(
    path: Union[os.PathLike, str], batch_size: int, patch_shape: Tuple[int, int], download: bool = False, **kwargs
) -> DataLoader:
    """Get the Brain Organoids dataloader for organoid segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_brain_organoids_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
