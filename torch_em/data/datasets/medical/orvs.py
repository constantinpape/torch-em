"""The ORVS dataset contains annotations for retinal vessel segmentation in fundus images.

It comprises 49 high-resolution fundus images (42 for training and 7 for testing) collected
at a clinic in Calgary, Canada, with vessel masks manually traced by a trained expert.

NOTE: The original repository (AbdullahSarhan/ICPRVessels) referenced by the publication has
been removed from GitHub. This module downloads the dataset from a public fork of it instead:
https://github.com/hitszsyl/ICPRVessels (no explicit license is stated in the repository).
This dataset is from the publication https://doi.org/10.48550/arXiv.2012.09250.
Please cite it if you use this dataset in your research.

NOTE: The labels are stored as JPEG images, so the (originally binary) vessel masks have lossy
compression artifacts near the mask boundaries. This module binarizes them with a fixed
intensity threshold when caching the labels to disk.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://github.com/hitszsyl/ICPRVessels/archive/421886be66a400d88ef6fe9c9e047d586486eee0.zip"
CHECKSUM = "45f8c7fc637c6e302d2cf6c71251aeb53d0a08f4411d5e60bd7ae6753acdb24b"

SPLIT_DIRS = {"train": "Train", "test": "Test"}


def get_orvs_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ORVS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the downloaded ORVS images and vessel annotations.
    """
    data_dir = os.path.join(
        path, "ICPRVessels-421886be66a400d88ef6fe9c9e047d586486eee0", "Vessels-Datasets", "ORVS"
    )
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "ICPRVessels.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_orvs_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"] = "train", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the ORVS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_orvs_data(path=path, download=download)

    if split not in SPLIT_DIRS:
        raise ValueError(f"'{split}' is not a valid split.")

    split_dir = os.path.join(data_dir, SPLIT_DIRS[split], "Original")
    image_paths = natsorted(glob(os.path.join(split_dir, "Images", "*.jpg")))

    label_dir = os.path.join(split_dir, "Labels_binary")
    os.makedirs(label_dir, exist_ok=True)

    gt_paths = []
    for image_path in tqdm(image_paths, desc="Preprocessing ORVS labels"):
        fname = os.path.splitext(os.path.basename(image_path))[0]
        gt_path = os.path.join(label_dir, f"{fname}.tif")
        gt_paths.append(gt_path)
        if os.path.exists(gt_path):
            continue

        label_path = os.path.join(split_dir, "Labels", f"{fname}.jpg")
        label = imageio.imread(label_path)
        label = (np.asarray(label) > 127).astype("uint8")
        if label.ndim == 3:
            label = label.max(axis=-1)
        imageio.imwrite(gt_path, label, compression="zlib")

    return image_paths, gt_paths


def get_orvs_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ORVS dataset for retinal vessel segmentation in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_orvs_paths(path, split, download)

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
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_orvs_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ORVS dataloader for retinal vessel segmentation in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_orvs_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
