"""The DenPAR dataset contains annotations for tooth segmentation in intraoral periapical (IOPA)
radiographs.

The dataset is from the publication https://doi.org/10.1038/s41597-025-05906-9. Please cite it
if you use this dataset in your research.

This module uses version 3 of the dataset (Zenodo record 16645076), which is openly downloadable.
Earlier versions (v1: 14181645, v2: 13998619) are restricted and require a Zenodo access request.

The dataset also provides bone-level annotations and keypoint (CEJ, APEX) annotations, which are
not exposed by this module; only the radiograph-wise (semantic) and tooth-wise (instance) tooth
segmentation masks are used here.
"""

import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/16645076/files/DenPAR%20Radiographs%20Dataset.zip"
CHECKSUM = "b9edb55020f2cb971ba771b4cf5e4b65c4abb4df957310bd2eccc83d5a08b072"

SPLITS = {"train": "Training", "val": "Validation", "test": "Testing"}


def get_denpar_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the DenPAR dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "denpar_radiographs_dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _rasterize_instances(image_path, tooth_mask_dir, preprocessed_path):
    image_id = Path(image_path).stem
    tooth_mask_paths = natsorted(glob(os.path.join(tooth_mask_dir, image_id, "*.png")))

    shape = imageio.imread(image_path).shape[:2]
    instances = np.zeros(shape, dtype="uint16")
    for i, tooth_mask_path in enumerate(tooth_mask_paths, start=1):
        mask = imageio.imread(tooth_mask_path) > 0
        instances[mask] = i

    imageio.imwrite(preprocessed_path, instances)


def get_denpar_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    label_choice: Literal["semantic", "instance"] = "semantic",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the DenPAR data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'train', 'val' or 'test'.
        label_choice: The choice of segmentation labels. Either 'semantic' (binary tooth mask,
            one mask per radiograph) or 'instance' (individual tooth instances, rasterized from
            the per-tooth masks into a single label map).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Please choose from {list(SPLITS.keys())}.")

    if label_choice not in ("semantic", "instance"):
        raise ValueError(f"'{label_choice}' is not a valid label choice. Please choose 'semantic' or 'instance'.")

    data_dir = get_denpar_data(path, download)
    split_dir = os.path.join(data_dir, SPLITS[split])

    image_paths = natsorted(glob(os.path.join(split_dir, "Images", "*.jpg")))

    if label_choice == "semantic":
        gt_dir = os.path.join(split_dir, "Masks (Radiograph-wise)")
        gt_paths = [os.path.join(gt_dir, f"{Path(p).stem}.png") for p in image_paths]
        image_paths = [p for p, g in zip(image_paths, gt_paths) if os.path.exists(g)]
        gt_paths = [g for g in gt_paths if os.path.exists(g)]

    else:
        tooth_mask_dir = os.path.join(split_dir, "Masks (Tooth-wise)")
        preprocessed_dir = os.path.join(split_dir, "preprocessed_instances")
        os.makedirs(preprocessed_dir, exist_ok=True)

        fimage_paths, gt_paths = [], []
        for image_path in image_paths:
            image_id = Path(image_path).stem
            if not os.path.exists(os.path.join(tooth_mask_dir, image_id)):
                continue

            gt_path = os.path.join(preprocessed_dir, f"{image_id}.tif")
            if not os.path.exists(gt_path):
                _rasterize_instances(image_path, tooth_mask_dir, gt_path)

            fimage_paths.append(image_path)
            gt_paths.append(gt_path)

        image_paths = fimage_paths

    return image_paths, gt_paths


def get_denpar_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    label_choice: Literal["semantic", "instance"] = "semantic",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DenPAR dataset for tooth segmentation in intraoral periapical radiographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        label_choice: The choice of segmentation labels. Either 'semantic' or 'instance'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_denpar_paths(path, split, label_choice, download)

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


def get_denpar_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    label_choice: Literal["semantic", "instance"] = "semantic",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DenPAR dataloader for tooth segmentation in intraoral periapical radiographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        label_choice: The choice of segmentation labels. Either 'semantic' or 'instance'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_denpar_dataset(path, patch_shape, split, label_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
