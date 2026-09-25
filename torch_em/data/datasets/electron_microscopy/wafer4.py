"""The Wafer4 dataset contains annotations for neuron segmentation
in serial section electron microscopy of the mouse medial entorhinal cortex.

The dataset is one volume of 125 x 1250 x 1250 voxels with a voxel level instance annotation of the
neurons. It was imaged with multi beam scanning electron microscopy at a resolution of 8 x 8 x 35
nanometer. The authors split it along the z axis, into 100 sections for training and 25 sections for
testing, and this loader follows that split.

NOTE: The volume covers layer six of the allocortex, which the common electron microscopy neuron
datasets do not cover.

The dataset is located at https://github.com/liuxy1103/CAD under the CC BY-NC 4.0 license.
This dataset is from the publication https://doi.org/10.1109/CVPR52733.2024.01056.
Please cite it if you use this dataset in your research.
"""

import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URLS = {
    "raw": "https://drive.usercontent.google.com/download?id=1l8Lhk-icIWyDb3fDt_2dDIvvx7S_9vLS&confirm=xxx",
    "labels": "https://drive.usercontent.google.com/download?id=1yyr3eo3-IQsEVvdZIgLsf_QrzaqXfUUX&confirm=xxx",
}

CHECKSUMS = {
    "raw": "5bb64ae54d5d89a501b6942a999a4210eb178671293b5053ad63e12211df603a",
    "labels": "7d3eab21447a0efe327b5192303cc8df8f74ca280eb04e020cbe7dc27beb59c4",
}

FILE_NAMES = {"raw": "wafer4_inputs.h5", "labels": "wafer4_labels.h5"}

N_SECTIONS = 125
N_TRAIN_SECTIONS = 100


def get_wafer4_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Wafer4 dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder that holds the data.
    """
    os.makedirs(path, exist_ok=True)

    for key, file_name in FILE_NAMES.items():
        file_path = os.path.join(path, file_name)
        if os.path.exists(file_path):
            continue
        util.download_source(file_path, URLS[key], download, CHECKSUMS[key])

    return path


def get_wafer4_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[str, str]:
    """Get paths to the Wafer4 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath for the image data.
        The filepath for the label data.
    """
    data_dir = get_wafer4_data(path, download)
    raw_path = os.path.join(data_dir, FILE_NAMES["raw"])
    label_path = os.path.join(data_dir, FILE_NAMES["labels"])
    return raw_path, label_path


def _get_split_roi(split: Optional[str]) -> Any:
    """Get the region of interest of a split. The authors split the volume along the z axis."""
    if split is None:
        return np.s_[:, :, :]
    if split == "train":
        return np.s_[:N_TRAIN_SECTIONS, :, :]
    if split == "test":
        return np.s_[N_TRAIN_SECTIONS:, :, :]
    raise ValueError(f"'{split}' is not a valid split. Choose 'train' or 'test', or None for the full volume.")


def get_wafer4_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    split: Optional[Literal["train", "test"]] = "train",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    rois: Optional[Dict[str, Any]] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Wafer4 dataset for the segmentation of neurons in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 3D patch shape to use for training.
        split: The data split. Either 'train' for the first 100 sections, 'test' for the last 25
            sections, or None for the full volume.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        rois: The region of interest to use. Overrides the region of interest of the split.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 3:
        raise ValueError(f"The Wafer4 patch shape must be three-dimensional, got {patch_shape}.")

    raw_path, label_path = get_wafer4_paths(path, download)
    roi = _get_split_roi(split) if rois is None else rois

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, offsets=offsets, boundaries=boundaries, binary=binary,
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_path,
        raw_key="main",
        label_paths=label_path,
        label_key="main",
        patch_shape=patch_shape,
        rois=roi,
        ndim=3,
        **kwargs,
    )


def get_wafer4_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    split: Optional[Literal["train", "test"]] = "train",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    rois: Optional[Dict[str, Any]] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the Wafer4 dataloader for the segmentation of neurons in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 3D patch shape to use for training.
        split: The data split. Either 'train' for the first 100 sections, 'test' for the last 25
            sections, or None for the full volume.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        rois: The region of interest to use. Overrides the region of interest of the split.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_wafer4_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        rois=rois,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
