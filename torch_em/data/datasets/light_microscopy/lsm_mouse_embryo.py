"""The LSM Mouse Embryo dataset contains annotations for tissue and cell segmentation
in light-sheet microscopy images of mouse embryos.

NOTE: The dataset only has semantic segmentation.

The dataset is from the publication https://doi.org/10.1109/ACCESS.2022.3210542.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Literal, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://www.dropbox.com/s/7zkk4j415ncfs47/LSM_Segmentation_Dataset.zip?dl=1"
CHECKSUM = None

TASKS = {
    "tissue": {"dir": "DAPI-Tissue", "mask_dir": "Mask"},
    "cells": {"dir": "DAPI-Cells", "mask_dir": "Mesen_Mask"},
    "proliferating_cells": {"dir": "PHH3-Cells", "mask_dir": "Mask"},
}

TASK_NAMES = list(TASKS.keys())
SPLITS = ["Training", "Validation", "Test"]
_SPLIT_MAPPING = {"train": "Training", "val": "Validation", "test": "Test"}


def _preprocess_masks(mask_dir, processed_dir):
    """Normalize masks to single-channel uint8 format.

    Some PHH3-Cells masks are stored as RGBA PNGs instead of binary masks.
    This function converts all masks to a consistent single-channel uint8 format.
    """
    os.makedirs(processed_dir, exist_ok=True)

    mask_paths = natsorted(glob(os.path.join(mask_dir, "*.png")))
    processed_paths = []
    for mask_path in mask_paths:
        fname = os.path.basename(mask_path)
        out_path = os.path.join(processed_dir, fname.replace(".png", ".tif"))
        processed_paths.append(out_path)

        if os.path.exists(out_path):
            continue

        mask = imageio.imread(mask_path)

        # Handle RGBA/RGB masks: convert to binary using the first channel.
        if mask.ndim == 3:
            mask = (mask[..., 0] > 0)

        mask = np.asarray(mask, dtype="uint8")
        imageio.imwrite(out_path, mask, compression="zlib")

    return processed_paths


def get_lsm_mouse_embryo_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the LSM Mouse Embryo dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data directory.
    """
    data_dir = os.path.join(path, "LSM_Segmentation_Dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "LSM_Segmentation_Dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_lsm_mouse_embryo_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"] = "train",
    task: Literal["tissue", "cells", "proliferating_cells"] = "tissue",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the LSM Mouse Embryo data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. One of 'train', 'val' or 'test'.
        task: The segmentation task. One of 'tissue' (3-class semantic segmentation of neural
            ectoderm and mesenchyme), 'cells' (binary cell segmentation in DAPI-stained images)
            or 'proliferating_cells' (binary segmentation of pHH3-stained proliferating cells).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    assert split in _SPLIT_MAPPING, f"'{split}' is not a valid split. Choose from {list(_SPLIT_MAPPING.keys())}."
    assert task in TASKS, f"'{task}' is not a valid task. Choose from {TASK_NAMES}."

    data_dir = get_lsm_mouse_embryo_data(path, download)
    split_name = _SPLIT_MAPPING[split]

    task_info = TASKS[task]
    image_dir = os.path.join(data_dir, task_info["dir"], split_name, "Original")
    mask_dir = os.path.join(data_dir, task_info["dir"], split_name, task_info["mask_dir"])

    image_paths = natsorted(glob(os.path.join(image_dir, "*.png")))
    assert len(image_paths) > 0, f"No images found in {image_dir}"

    # Preprocess masks to ensure consistent single-channel format.
    processed_dir = os.path.join(path, "processed_masks", task, split_name)
    if not os.path.exists(processed_dir) or len(glob(os.path.join(processed_dir, "*.tif"))) == 0:
        seg_paths = _preprocess_masks(mask_dir, processed_dir)
    else:
        seg_paths = natsorted(glob(os.path.join(processed_dir, "*.tif")))

    assert len(image_paths) == len(seg_paths), \
        f"Mismatch: {len(image_paths)} images vs {len(seg_paths)} masks for {task}/{split_name}"

    return image_paths, seg_paths


def get_lsm_mouse_embryo_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"] = "train",
    task: Literal["tissue", "cells", "proliferating_cells"] = "tissue",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LSM Mouse Embryo dataset for tissue and cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train', 'val' or 'test'.
        task: The segmentation task. One of 'tissue' (3-class semantic segmentation of neural
            ectoderm and mesenchyme), 'cells' (binary cell segmentation in DAPI-stained images)
            or 'proliferating_cells' (binary segmentation of pHH3-stained proliferating cells).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, seg_paths = get_lsm_mouse_embryo_paths(path, split, task, download)

    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=seg_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        ndim=2,
        **kwargs
    )


def get_lsm_mouse_embryo_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"] = "train",
    task: Literal["tissue", "cells", "proliferating_cells"] = "tissue",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LSM Mouse Embryo dataloader for tissue and cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train', 'val' or 'test'.
        task: The segmentation task. One of 'tissue' (3-class semantic segmentation of neural
            ectoderm and mesenchyme), 'cells' (binary cell segmentation in DAPI-stained images)
            or 'proliferating_cells' (binary segmentation of pHH3-stained proliferating cells).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lsm_mouse_embryo_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        task=task,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
