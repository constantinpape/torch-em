"""The SIIM ACR dataset contains annotations for pneumothorax segmentation in
chest X-Rays.

This dataset is located at https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks/data.
The dataset is from the "SIIM-ACR Pneumothorax Segmentation" competition:
https://kaggle.com/competitions/siim-acr-pneumothorax-segmentation.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


KAGGLE_DATASET_NAME = "vbookshelf/pneumothorax-chest-xray-images-and-masks"
CHECKSUM = "1ade68d31adb996c531bb686fb9d02fe11876ddf6f25594ab725e18c69d81538"


def get_siim_acr_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the SIIM ACR dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "siim-acr-pneumothorax")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name=KAGGLE_DATASET_NAME, download=download)

    zip_path = os.path.join(path, "pneumothorax-chest-xray-images-and-masks.zip")
    util._check_checksum(path=zip_path, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_siim_acr_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'val', 'test'], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the SIIM ACR data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_siim_acr_data(path=path, download=download)

    if split == "test":
        image_paths = natsorted(glob(os.path.join(data_dir, "png_images", f"*_{split}_*.png")))
        gt_paths = natsorted(glob(os.path.join(data_dir, "png_masks", f"*_{split}_*.png")))
    else:
        image_paths = natsorted(glob(os.path.join(data_dir, "png_images", "*_train_*.png")))
        gt_paths = natsorted(glob(os.path.join(data_dir, "png_masks", "*_train_*.png")))

        if split == "train":
            image_paths, gt_paths = image_paths[:100], gt_paths[:100]
        elif split == "val":
            image_paths, gt_paths = image_paths[100:], gt_paths[100:]
        else:
            raise ValueError(f"'{split}' is not a valid split.")

    return image_paths, gt_paths


def get_siim_acr_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SIIM ACR dataset for pneumothorax segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_siim_acr_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )
    dataset.max_sampling_attempts = 5000

    return dataset


def get_siim_acr_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SIIM ACR dataloader for pneumothorax segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_siim_acr_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
