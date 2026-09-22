"""The PUWF-AV dataset contains annotations for artery-vein segmentation in pediatric
ultra-widefield (UWF) fundus images.

For the class labels: red represents arteries, blue represents veins, green represents artery-vein
crossings, and white represents vessels of uncertain classification.

NOTE: The dataset does not ship an official train / val / test split, so `get_puwf_av_paths` returns
all 60 images.

This dataset is from the publication https://doi.org/10.1038/s41597-026-08342-5.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/65105883"
CHECKSUM = "d8b9a060dde125e767f999de7cbfc8d49cfac584e22611d915c40e9ceb21a054"


def _process_labels(data_dir):
    label_paths = glob(os.path.join(data_dir, "annotation", "*.png"))
    for label_path in label_paths:
        labels = imageio.imread(label_path)

        neu_labels = np.zeros(labels.shape[:2])
        neu_labels[np.all(labels == (255, 0, 0), axis=-1)] = 1  # red are arteries.
        neu_labels[np.all(labels == (0, 0, 255), axis=-1)] = 2  # blue are veins.
        neu_labels[np.all(labels == (0, 255, 0), axis=-1)] = 3  # green are overlaps.
        neu_labels[np.all(labels == (255, 255, 255), axis=-1)] = 4  # white are unknown.

        imageio.imwrite(Path(label_path).with_suffix(".tif"), neu_labels, compression="zlib")

        os.remove(label_path)


def get_puwf_av_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PUWF-AV dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "PUWF-AV")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "PUWF-AV.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    _process_labels(data_dir)

    return data_dir


def get_puwf_av_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the PUWF-AV data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_puwf_av_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "images", "*.png")))
    label_paths = natsorted(glob(os.path.join(data_dir, "annotation", "*.tif")))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_puwf_av_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PUWF-AV dataset for artery-vein segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_puwf_av_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        ndim=2,
        **kwargs
    )


def get_puwf_av_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PUWF-AV dataloader for artery-vein segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_puwf_av_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
