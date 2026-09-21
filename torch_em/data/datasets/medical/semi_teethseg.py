"""The Semi-TeethSeg (STS-Tooth) dataset contains annotations for tooth segmentation in panoramic
dental X-rays (PXI) and dental CBCT scans.

This is a multi-modal dataset for semi-supervised learning: alongside the labeled images, it also
ships large unlabeled sets of panoramic X-rays and CBCT scans (not covered by this module, as they
have no ground truth). This module only covers the labeled panoramic X-ray subset (STS-2D-Tooth),
which has binary tooth segmentation masks for 900 adult and pediatric panoramic radiographs. The
dataset also ships a labeled CBCT subset (STS-3D-Tooth), which is not covered by this module.

The dataset was curated for the MICCAI 2023 and 2024 Semi-supervised Teeth Segmentation (STS)
challenges (https://sts-challenge.github.io/miccai2024/index.html) and is hosted on Zenodo at
https://doi.org/10.5281/zenodo.10597292, distributed under the CC BY 4.0 license.

The dataset is from the publication https://doi.org/10.1038/s41597-024-04306-9.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Literal, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://zenodo.org/records/10597292/files"

# The dataset is distributed as a 15-part split zip archive (SD-Tooth.zip.001 - .015, ~32GB total).
# The parts are joined and unzipped once to extract the full 'SD-Tooth' folder. Only part 001 (which
# already contains the full 'STS-2D-Tooth' subtree used by this module) has been downloaded and
# verified so far, so only its checksum is set here; the others are left unverified.
CHECKSUMS = {
    "001": "916e9c9e72790c4bdf0884d012455c43821b2737bb27915e812dc20c2ff21cf4",
    "002": None,
    "003": None,
    "004": None,
    "005": None,
    "006": None,
    "007": None,
    "008": None,
    "009": None,
    "010": None,
    "011": None,
    "012": None,
    "013": None,
    "014": None,
    "015": None,
}


def get_semi_teethseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Semi-TeethSeg dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "SD-Tooth")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    part_paths = []
    for part_id, checksum in CHECKSUMS.items():
        part_path = os.path.join(path, f"SD-Tooth.zip.{part_id}")
        url = f"{BASE_URL}/SD-Tooth.zip.{part_id}?download=1"
        util.download_source(path=part_path, url=url, download=download, checksum=checksum)
        part_paths.append(part_path)

    joined_zip_path = os.path.join(path, "SD-Tooth.zip")
    if not os.path.exists(joined_zip_path):
        with open(joined_zip_path, "wb") as dst:
            for part_path in part_paths:
                with open(part_path, "rb") as src:
                    while True:
                        chunk = src.read(1024 * 1024 * 64)
                        if not chunk:
                            break
                        dst.write(chunk)

    util.unzip(zip_path=joined_zip_path, dst=path, remove=False)

    return data_dir


def get_semi_teethseg_paths(
    path: Union[os.PathLike, str], split: Literal["adult", "child"] = "adult", download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Semi-TeethSeg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'adult' (A-PXI) or 'child' (C-PXI).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_semi_teethseg_data(path, download)

    modality_dir = "A-PXI" if split == "adult" else "C-PXI"
    base_dir = os.path.join(data_dir, "STS-2D-Tooth", modality_dir, "Labeled")

    image_paths = natsorted(glob(os.path.join(base_dir, "Image", "*.png")))
    raw_gt_paths = natsorted(glob(os.path.join(base_dir, "Mask", "*.png")))

    assert len(image_paths) == len(raw_gt_paths) and len(image_paths) > 0

    neu_gt_dir = os.path.join(data_dir, "preprocessed", modality_dir)
    os.makedirs(neu_gt_dir, exist_ok=True)

    gt_paths = []
    for raw_gt_path in tqdm(raw_gt_paths, desc="Preprocessing labels"):
        gt_path = os.path.join(neu_gt_dir, f"{Path(raw_gt_path).stem}.tif")
        gt_paths.append(gt_path)
        if os.path.exists(gt_path):
            continue

        # the original masks are stored as single-bit (boolean) images, i.e. non-zero pixels
        # correspond to teeth. we binarize them into a uint8 (0, 1) label map.
        raw_gt = imageio.imread(raw_gt_path)
        binary_gt = (raw_gt > 0).astype(np.uint8)
        imageio.imwrite(gt_path, binary_gt)

    return image_paths, gt_paths


def get_semi_teethseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["adult", "child"] = "adult",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Semi-TeethSeg dataset for tooth segmentation in panoramic dental radiographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'adult' (A-PXI) or 'child' (C-PXI).
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_semi_teethseg_paths(path, split, download)

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
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_semi_teethseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["adult", "child"] = "adult",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Semi-TeethSeg dataloader for tooth segmentation in panoramic dental radiographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'adult' (A-PXI) or 'child' (C-PXI).
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_semi_teethseg_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
