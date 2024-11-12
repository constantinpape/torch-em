"""The DRIVE dataset contains annotations for retinal vessel segmentation in
fundus images.

This dataset is from the "DRIVE" challenge: https://drive.grand-challenge.org/.
The dataset is from the publication https://doi.org/10.1109/TMI.2004.825627.
Please cite them if you use this dataset for your research.
"""

import os
from glob import glob
from pathlib import Path
from typing import Union, Tuple, Literal, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = {
    "train": "https://www.dropbox.com/sh/z4hbbzqai0ilqht/AADp_8oefNFs2bjC2kzl2_Fqa/training.zip?dl=1",
    "test": "https://www.dropbox.com/sh/z4hbbzqai0ilqht/AABuUJQJ5yG5oCuziYzYu8jWa/test.zip?dl=1"
}

CHECKSUM = {
    "train": "7101e19598e2b7aacdbd5e6e7575057b9154a4aaec043e0f4e28902bf4e2e209",
    "test": "d76c95c98a0353487ffb63b3bb2663c00ed1fde7d8fdfd8c3282c6e310a02731"
}


def get_drive_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the DRIVE dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "training")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "training.zip")
    util.download_source_gdrive(
        path=zip_path, url=URL["train"], download=download, checksum=CHECKSUM["train"], download_type="zip",
    )
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_drive_ground_truth(data_dir):
    gt_paths = sorted(glob(os.path.join(data_dir, "1st_manual", "*.gif")))

    neu_gt_dir = os.path.join(data_dir, "gt")
    if os.path.exists(neu_gt_dir):
        return sorted(glob(os.path.join(neu_gt_dir, "*.tif")))
    else:
        os.makedirs(neu_gt_dir, exist_ok=True)

    neu_gt_paths = []
    for gt_path in gt_paths:
        gt = imageio.imread(gt_path).squeeze()
        neu_gt_path = os.path.join(
            neu_gt_dir, Path(os.path.split(gt_path)[-1]).with_suffix(".tif")
        )
        imageio.imwrite(neu_gt_path, (gt > 0).astype("uint8"))
        neu_gt_paths.append(neu_gt_path)

    return neu_gt_paths


def get_drive_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'val', 'test'], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the DRIVE data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_drive_data(path=path, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "images", "*.tif")))
    gt_paths = _get_drive_ground_truth(data_dir)

    if split == "train":
        image_paths, gt_paths = image_paths[:10], gt_paths[:10]
    elif split == "val":
        image_paths, gt_paths = image_paths[10:14], gt_paths[10:14]
    elif split == "test":
        image_paths, gt_paths = image_paths[14:], gt_paths[14:]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    return image_paths, gt_paths


def get_drive_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DRIVE dataset for segmentation of retinal blood vessels in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_drive_paths(path=path, split=split, download=download)

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


def get_drive_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: str,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DRIVE dataloader for segmentation of retinal blood vessels in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_drive_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
