"""FAR-POLYP-SEG is a prospectively collected colonoscopy dataset for colorectal polyp
segmentation, acquired at Farhikhtegan Hospital, Islamic Azad University, Tehran, Iran
(February-December 2025), with clinician-created and gastroenterologist-reviewed
pixel-level masks.

The dataset contains 8,181 RGB frames from 455 patients: 432 polyp-positive frames with
expert-verified binary masks and 7,749 normal colonic mucosa frames (without polyps or
masks). This module exposes only the 432 polyp-positive image-mask pairs, as the
normal-mucosa frames have no corresponding annotations to train a segmentation model on.

The dataset is located at https://doi.org/10.5281/zenodo.20284781 and is licensed under
CC-BY-4.0.

This dataset is from the publication https://doi.org/10.1007/s10278-026-02268-5.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/20284781/files/Dataset.zip"
CHECKSUM = "026a7b1e2ce9407e77b73373e999afcdbcbac776c5ac84c7944acf6285ed994d"


def get_far_polyp_seg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the FAR-POLYP-SEG data.

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

    zip_path = os.path.join(path, "Dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_far_polyp_seg_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the FAR-POLYP-SEG data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_far_polyp_seg_data(path, download)

    image_paths = natsorted(glob(os.path.join(data_dir, "Patient*", "Polyp", "*.jpg")))

    # The masks are lossily JPEG-compressed grayscale images (background near 0, foreground
    # near 255), not the clean binary masks 'ImageCollectionDataset' expects. They are
    # binarized once here and cached as '.tif' files next to the original masks.
    gt_paths = []
    for image_path in tqdm(image_paths, desc="Preprocessing FAR-POLYP-SEG masks"):
        patient_dir = os.path.dirname(os.path.dirname(image_path))
        mask_path = os.path.join(patient_dir, "BinaryMask", os.path.basename(image_path))
        neu_gt_path = os.path.join(patient_dir, "BinaryMask", f"{Path(mask_path).stem}.tif")
        gt_paths.append(neu_gt_path)
        if os.path.exists(neu_gt_path):
            continue

        mask = imageio.imread(mask_path)
        if mask.ndim == 3:
            mask = np.mean(mask, axis=-1)
        mask = (mask > 128).astype("uint8")
        imageio.imwrite(neu_gt_path, mask, compression="zlib")

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0, (
        "No image-mask pairs were found. The expected per-patient 'Polyp' / 'BinaryMask' folder layout "
        "may not match the actual structure of the downloaded data. Please inspect the data at "
        f"'{data_dir}'."
    )

    return image_paths, gt_paths


def get_far_polyp_seg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the FAR-POLYP-SEG dataset for polyp segmentation in colonoscopy images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_far_polyp_seg_paths(path, download)

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


def get_far_polyp_seg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the FAR-POLYP-SEG dataloader for polyp segmentation in colonoscopy images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_far_polyp_seg_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
