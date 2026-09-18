"""BAGLS-VF extends the BAGLS glottis segmentation benchmark with pixel-wise annotations
of the left vocal fold, right vocal fold and glottal area (3 foreground classes), derived
from high-speed videolaryngoscopy recordings. Masks are RGB-coded, with a distinct color
per anatomical structure (encoded here as separate label values).

The dataset is located at https://doi.org/10.5281/zenodo.19593658.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from pathlib import Path
from typing import Union, Tuple, List, Literal

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "train": "https://zenodo.org/records/19593658/files/BAGLS-VF_train.zip",
    "test": "https://zenodo.org/records/19593658/files/BAGLS-VF_test.zip",
}

CHECKSUMS = {
    "train": "accc5bc0acf920f344df5efb5b159fd15ad1a0941536542f65080a9f91b94c90",
    "test": "0876ae6461d5142b08338c4e45b9c96a4aa7642bebab4ce8d4a8508585b0907f",
}

LABEL_MAP = {
    (0, 0, 0): 0,  # background
    (0, 0, 255): 1,  # vocal fold / glottis structure (blue)
    (0, 255, 0): 2,  # vocal fold / glottis structure (green)
    (255, 0, 0): 3,  # vocal fold / glottis structure (red)
}


def get_bagls_vf_data(path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False) -> str:
    """Download the BAGLS-VF dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, f"BAGLS-VF_{split}")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"BAGLS-VF_{split}.zip")
    util.download_source(path=zip_path, url=URLS[split], download=download, checksum=CHECKSUMS[split])
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_bagls_vf_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the BAGLS-VF data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_bagls_vf_data(path=path, split=split, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, f"{split}_images", "*.png")))
    mask_dir = os.path.join(data_dir, f"{split}_masks")

    neu_gt_dir = os.path.join(data_dir, "preprocessed_masks")
    os.makedirs(neu_gt_dir, exist_ok=True)

    gt_paths = []
    for image_path in image_paths:
        fname = Path(image_path).stem
        mask_path = os.path.join(mask_dir, f"{fname}.png")
        neu_gt_path = os.path.join(neu_gt_dir, f"{fname}.tif")
        gt_paths.append(neu_gt_path)

        if os.path.exists(neu_gt_path):
            continue

        mask = imageio.imread(mask_path)[..., :3]
        instances = np.zeros(mask.shape[:2], dtype="uint8")
        for color, label in LABEL_MAP.items():
            if label == 0:
                continue
            binary_map = (mask == color).all(axis=-1)
            instances[binary_map] = label

        imageio.imwrite(neu_gt_path, instances, compression="zlib")

    return image_paths, gt_paths


def get_bagls_vf_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BAGLS-VF dataset for vocal fold and glottal area segmentation.

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
    image_paths, gt_paths = get_bagls_vf_paths(path, split, download)

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


def get_bagls_vf_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BAGLS-VF dataloader for vocal fold and glottal area segmentation.

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
    dataset = get_bagls_vf_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
