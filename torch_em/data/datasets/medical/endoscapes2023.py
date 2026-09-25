"""The Endoscapes2023 dataset contains annotations for segmentation of hepatocystic anatomy and
surgical tools in laparoscopic cholecystectomy images.

This is the segmentation subset (Endoscapes-Seg50) of the Endoscapes2023 dataset: 493 frames from
50 videos, annotated with semantic and instance segmentation masks for 6 anatomical structures / a
tool class. The full dataset additionally contains bounding box (Endoscapes-BBox201) and Critical
View of Safety (CVS) classification annotations (Endoscapes-CVS201) for many more frames, but no
pixel-level masks for those; they are not covered by this module.

The dataset is located at https://github.com/CAMMA-public/Endoscapes and downloaded from
https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip (~6.3 GB; the segmentation
subset cannot be downloaded on its own, the full archive is always fetched). It is licensed under
CC BY-NC-SA 4.0 (non-commercial research use only).

This dataset is from the publication https://doi.org/10.48550/arXiv.2312.12429.
Please additionally cite https://doi.org/10.48550/arXiv.2112.13815 if you use the segmentation
annotations (Endoscapes-Seg50) in a publication.
"""

import os
import shutil
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip"
CHECKSUM = "0574b11a82779a1a0a783e2083627b1140de096e8b7172903e35afb583d75862"

CLASSES = ["background", "cystic_plate", "calot_triangle", "cystic_artery", "cystic_duct", "gallbladder", "tool"]
"""The classes of the Endoscapes2023 segmentation masks, in order of their label id (see `seg_label_map.txt`
in the downloaded data)."""

SPLIT_DIRS = {"train": "train_seg", "val": "val_seg", "test": "test_seg"}


def get_endoscapes2023_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Endoscapes2023 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "endoscapes")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "endoscapes.zip")
    print("Downloading the Endoscapes2023 data. This is a ~6.3 GB archive, it might take a while.")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_endoscapes2023_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Endoscapes2023 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLIT_DIRS:
        raise ValueError(f"'{split}' is not a valid split.")

    data_dir = get_endoscapes2023_data(path, download)
    image_dir = os.path.join(data_dir, SPLIT_DIRS[split])

    ppdir = os.path.join(data_dir, "preprocessed", split)
    prepped_images = os.path.join(ppdir, "images")
    prepped_masks = os.path.join(ppdir, "masks")
    if os.path.exists(prepped_images) and os.path.exists(prepped_masks):
        return natsorted(glob(os.path.join(prepped_images, "*.jpg"))), natsorted(glob(os.path.join(prepped_masks, "*.tif")))  # noqa

    os.makedirs(prepped_images, exist_ok=True)
    os.makedirs(prepped_masks, exist_ok=True)

    # The semantic masks live in one shared 'semseg' folder for all splits; only the ones whose
    # frame id also exists in this split's image folder belong to this split.
    mask_paths = natsorted(glob(os.path.join(data_dir, "semseg", "*.png")))

    image_paths, gt_paths = [], []
    for mask_path in mask_paths:
        frame_id = Path(mask_path).stem
        src_image_path = os.path.join(image_dir, f"{frame_id}.jpg")
        if not os.path.exists(src_image_path):
            continue  # This mask belongs to a different split.

        dst_image_path = os.path.join(prepped_images, f"{frame_id}.jpg")
        dst_mask_path = os.path.join(prepped_masks, f"{frame_id}.tif")

        image_paths.append(dst_image_path)
        gt_paths.append(dst_mask_path)

        if os.path.exists(dst_image_path) and os.path.exists(dst_mask_path):
            continue

        mask = imageio.imread(mask_path)
        # Clean up rare annotation artifacts found in the raw masks: pixel value 255 marks
        # unlabeled / ambiguous regions (present in about a quarter of the masks), and a single
        # mask has a stray value of 7, which is outside the valid [0, 6] label range. Both are
        # mapped back to the background class.
        mask[(mask == 255) | (mask > (len(CLASSES) - 1))] = 0

        shutil.copy(src_image_path, dst_image_path)
        imageio.imwrite(dst_mask_path, mask, compression="zlib")

    assert image_paths and len(image_paths) == len(gt_paths)
    return image_paths, gt_paths


def get_endoscapes2023_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Endoscapes2023 dataset for anatomy and tool segmentation in laparoscopic cholecystectomy images.

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
    image_paths, gt_paths = get_endoscapes2023_paths(path, split, download)

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


def get_endoscapes2023_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Endoscapes2023 dataloader for anatomy and tool segmentation in laparoscopic cholecystectomy images.

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
    dataset = get_endoscapes2023_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
