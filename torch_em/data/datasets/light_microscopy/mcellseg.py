"""The mCellSeg dataset contains expert-annotated microscopy images for cell instance segmentation.

It contains 200 annotated 2D images from two human cell lines (HEK-293T and HUVEC),
acquired with differential interference contrast (DIC) and fluorescence microscopy.
Each image has a paired instance segmentation mask (0 = background, unique integer per cell).
A further 100 unannotated images are included for semi-supervised learning (not used here).

This dataset is from the publication:
https://doi.org/10.1016/j.cmpb.2026.108919
Please cite it if you use this dataset for a publication.

The data is available at https://doi.org/10.5281/zenodo.20174259.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Optional, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em
from .. import util


URL = "https://zenodo.org/records/20174259/files/mCellSeg.zip?download=1"
CHECKSUM = "55fec21acab10a78837718431f21f74e87e0777ebd5907ea9ef8a57a8a197217"


def get_mcellseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the mCellSeg dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Path to the folder containing the downloaded data.
    """
    data_dir = os.path.join(str(path), "mCellSeg")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(str(path), exist_ok=True)
    zip_path = os.path.join(str(path), "mCellSeg.zip")
    util.download_source(zip_path, URL, download, checksum=CHECKSUM)
    util.unzip(zip_path, str(path), remove=True)

    return data_dir


def get_mcellseg_paths(
    path: Union[os.PathLike, str],
    val_fraction: Optional[float] = None,
    split: Optional[str] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the mCellSeg image and mask files.

    Only the 200 images that have corresponding instance masks are returned.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        val_fraction: The fraction of data to use for validation. If None, all data is returned.
        split: The split to use, either "train" or "val". Required if val_fraction is set.
        download: Whether to download the data if it is not present.

    Returns:
        Tuple of (raw image paths, label mask paths).
    """
    data_dir = get_mcellseg_data(path, download)

    mask_paths = natsorted(glob(os.path.join(data_dir, "labeled", "masks", "*.tif")))
    raw_paths = []
    valid_mask_paths = []
    for mask_path in mask_paths:
        mask_name = os.path.basename(mask_path)
        img_name = mask_name.replace("_mask.tif", ".tif")
        img_path = os.path.join(data_dir, "labeled", "images", img_name)
        if os.path.exists(img_path):
            raw_paths.append(img_path)
            valid_mask_paths.append(mask_path)

    if val_fraction is not None:
        assert split in ("train", "val"), f"'split' must be 'train' or 'val', got '{split}'."
        n_val = max(1, int(len(raw_paths) * val_fraction))
        if split == "train":
            raw_paths = raw_paths[n_val:]
            valid_mask_paths = valid_mask_paths[n_val:]
        else:
            raw_paths = raw_paths[:n_val]
            valid_mask_paths = valid_mask_paths[:n_val]

    return raw_paths, valid_mask_paths


def get_mcellseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    val_fraction: Optional[float] = None,
    split: Optional[str] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs,
) -> Dataset:
    """Get the mCellSeg dataset for cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape (H, W) to use for training.
        val_fraction: The fraction of data to use for validation.
        split: The split to use, either "train" or "val". Required if val_fraction is set.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert sum((offsets is not None, boundaries, binary)) <= 1, f"{offsets}, {boundaries}, {binary}"

    raw_paths, label_paths = get_mcellseg_paths(path, val_fraction, split, download)

    if offsets is not None:
        label_transform = torch_em.transform.label.AffinityTransform(
            offsets=offsets, ignore_label=None, add_binary_target=True, add_mask=True
        )
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = util.update_kwargs(kwargs, "label_transform2", label_transform, msg=msg)
    elif boundaries:
        label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)
        msg = "Boundaries is set to True, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = util.update_kwargs(kwargs, "label_transform", label_transform, msg=msg)
    elif binary:
        label_transform = torch_em.transform.label.labels_to_binary
        msg = "Binary is set to True, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = util.update_kwargs(kwargs, "label_transform", label_transform, msg=msg)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", False)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_mcellseg_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    val_fraction: Optional[float] = None,
    split: Optional[str] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for cell instance segmentation in mCellSeg.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape (H, W) to use for training.
        batch_size: The batch size for training.
        val_fraction: The fraction of data to use for validation.
        split: The split to use, either "train" or "val". Required if val_fraction is set.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mcellseg_dataset(
        path, patch_shape, val_fraction=val_fraction, split=split, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary, **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
