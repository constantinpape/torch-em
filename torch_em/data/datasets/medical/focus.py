"""FOCUS is the Four-chamber Ultrasound Image Dataset for Fetal Cardiac Biometric
Measurement, with annotations for fetal heart and thorax segmentation in four-chamber
view ultrasound images, used e.g. to estimate the cardiothoracic diameter ratio.

The dataset is located at https://zenodo.org/records/14597550 (CC BY 4.0).
This dataset is from Zenodo, with DOI https://doi.org/10.5281/zenodo.14597550.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from typing import Union, Tuple, List, Literal

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/14597550/files/FOCUS-dataset.zip"
CHECKSUM = "625ee59d9d8adfb946790f03bd04e6342ad2a12499a0bed3ec02b65ec35369b8"

SPLIT_FOLDERS = {"train": "training", "val": "validation", "test": "testing"}


def get_focus_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the FOCUS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the downloaded images and segmentation masks.
    """
    if os.path.exists(os.path.join(path, "training")):
        return path

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "focus.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return path


def get_focus_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the FOCUS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLIT_FOLDERS:
        raise ValueError(f"'{split}' is not a supported split. Choose one of {list(SPLIT_FOLDERS.keys())}.")

    data_dir = get_focus_data(path=path, download=download)
    split_dir = os.path.join(data_dir, SPLIT_FOLDERS[split])

    image_paths = sorted(glob(os.path.join(split_dir, "images", "*.png")))

    label_dir = os.path.join(split_dir, "annfiles_semantic")
    os.makedirs(label_dir, exist_ok=True)

    gt_paths = []
    for image_path in tqdm(image_paths, desc=f"Preprocessing FOCUS '{split}' labels"):
        fname = os.path.splitext(os.path.basename(image_path))[0]
        gt_path = os.path.join(label_dir, f"{fname}.tif")
        gt_paths.append(gt_path)
        if os.path.exists(gt_path):
            continue

        thorax = imageio.imread(os.path.join(split_dir, "annfiles_mask", f"{fname}-thorax.png"))
        cardiac = imageio.imread(os.path.join(split_dir, "annfiles_mask", f"{fname}-cardiac.png"))
        if thorax.ndim == 3:
            thorax = thorax[..., 0]
        if cardiac.ndim == 3:
            cardiac = cardiac[..., 0]

        label = np.zeros(thorax.shape, dtype="uint8")
        label[thorax > 127] = 1
        label[cardiac > 127] = 2

        imageio.imwrite(gt_path, label, compression="zlib")

    return image_paths, gt_paths


def get_focus_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the FOCUS dataset for fetal cardiac and thorax segmentation.

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
    image_paths, gt_paths = get_focus_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
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


def get_focus_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the FOCUS dataloader for fetal cardiac and thorax segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_focus_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
