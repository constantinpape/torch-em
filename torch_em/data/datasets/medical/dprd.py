"""DPRD is a dataset for caries segmentation in children's dental panoramic radiographs.

This dataset is part of the "Children's Dental Panoramic Radiographs Dataset", which is hosted on
figshare at https://doi.org/10.6084/m9.figshare.21621705.v1 (part of the collection
https://doi.org/10.6084/m9.figshare.c.6317013.v1) and distributed under the CC0 license. This module
only makes use of the "Children's dental caries segmentation dataset" subset, which is the part of
the archive with pixel-level segmentation masks for dental caries. The raw masks are RGB images
with black background and a fixed color marking the caries region; this module collapses them to
a single-channel binary label map, where 0 is background and 1 marks caries.

The dataset is from the publication https://doi.org/10.1038/s41597-023-02237-5.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/38322366"
CHECKSUM = "2e41d862a0787828d659cfd13035e0fbaeb8687995dbfe29a84fdeac09b83945"

ZIP_SUBDIR = "Children's dental caries segmentation dataset"


def get_dprd_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the DPRD dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, ZIP_SUBDIR)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "Dental_dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)

    import zipfile
    with zipfile.ZipFile(zip_path) as f:
        members = [m for m in f.namelist() if m.startswith(f"{ZIP_SUBDIR}/")]
        f.extractall(path, members=members)
    os.remove(zip_path)

    return data_dir


def get_dprd_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the DPRD data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in ("train", "test"):
        raise ValueError(f"'{split}' is not a valid split. Please choose either 'train' or 'test'.")

    data_dir = get_dprd_data(path, download)

    split_dir = "Train" if split == "train" else "Test"
    image_paths = natsorted(glob(os.path.join(data_dir, split_dir, "images", "*.png")))
    raw_gt_paths = natsorted(glob(os.path.join(data_dir, split_dir, "mask", "*.png")))

    assert len(image_paths) == len(raw_gt_paths) and len(image_paths) > 0

    neu_gt_dir = os.path.join(data_dir, "preprocessed", split)
    os.makedirs(neu_gt_dir, exist_ok=True)

    gt_paths = []
    for raw_gt_path in tqdm(raw_gt_paths, desc="Preprocessing labels"):
        gt_path = os.path.join(neu_gt_dir, f"{Path(raw_gt_path).stem}.tif")
        gt_paths.append(gt_path)
        if os.path.exists(gt_path):
            continue

        # The raw masks are RGB images with black background and a fixed color (53, 119, 181)
        # marking the caries region. We collapse this to a single-channel binary label map,
        # where 0 is background and 1 marks caries.
        raw_gt = imageio.imread(raw_gt_path)
        binary_gt = (raw_gt.sum(axis=-1) > 0).astype(np.uint8)
        imageio.imwrite(gt_path, binary_gt)

    return image_paths, gt_paths


def get_dprd_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DPRD dataset for caries segmentation in panoramic dental radiographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'test'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_dprd_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_dprd_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DPRD dataloader for caries segmentation in panoramic dental radiographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'test'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_dprd_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
