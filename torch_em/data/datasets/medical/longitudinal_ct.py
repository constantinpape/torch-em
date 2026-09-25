"""Longitudinal-CT is a dataset of paired baseline and follow-up whole-body CT studies for longitudinal
tumor lesion segmentation and tracking in metastatic melanoma.

The dataset consists of 600 CT studies (300 patients, each with a baseline and a follow-up scan acquired
during systemic therapy) from the University Hospital Tubingen, with 7,182 manually segmented lesions in
total (4,079 at baseline, 3,103 at follow-up). Each CT is split per body region into one or more sub-volumes.
For every case and body region, the release provides: the baseline CT and its lesion mask, the follow-up CT
and its lesion mask, per-timepoint lesion center-of-gravity annotations (JSON) and lesion metadata (CSV) that
capture the longitudinal correspondence between baseline and follow-up lesions (eg. persistence, regression,
merging, new appearance). This module exposes the basic lesion segmentation task (CT -> binary lesion mask)
for both timepoints; the additional per-lesion correspondence metadata (JSON / CSV files) is not consumed by
`get_longitudinal_ct_dataset` / `get_longitudinal_ct_loader`, but remains available on disk next to the images.

The dataset is located at https://fdat.uni-tuebingen.de/records/qwsry-7t837 (DOI: 10.57754/FDAT.qwsry-7t837),
openly downloadable without registration. It is licensed under CC BY-NC 4.0.

This dataset is from the publication https://doi.org/10.1038/s41597-026-07466-y.
Please cite it if you use this dataset in your research.
"""

import os
import re
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://fdat.uni-tuebingen.de/api/records/qwsry-7t837/files/Longitudinal-CT.zip/content"
CHECKSUM = "ae361c9f163ab78d9f32cf23d20cf2b8af2d23d1a59d1a575d443edea32d4f89"


def get_longitudinal_ct_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Longitudinal-CT dataset.

    NOTE: This is a large dataset (~57 GB, distributed as a single zip archive).

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(os.path.join(data_dir, "inputsTr")) and os.path.exists(os.path.join(data_dir, "targetsTr")):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "Longitudinal-CT.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir, remove=False)

    return data_dir


def get_longitudinal_ct_paths(
    path: Union[os.PathLike, str],
    timepoint: Literal["baseline", "followup", "both"] = "both",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Longitudinal-CT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        timepoint: The choice of timepoint. One of 'baseline', 'followup' or 'both' (default).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if timepoint not in ("baseline", "followup", "both"):
        raise ValueError(f"'{timepoint}' is not a valid timepoint. Choose one of 'baseline', 'followup', 'both'.")

    data_dir = get_longitudinal_ct_data(path, download)
    inputs_dir = os.path.join(data_dir, "inputsTr")
    targets_dir = os.path.join(data_dir, "targetsTr")

    raw_paths, label_paths = [], []
    if timepoint in ("baseline", "both"):
        for image_path in natsorted(glob(os.path.join(inputs_dir, "*_BL_img_BL_img_*.nii.gz"))):
            label_path = re.sub(r"_BL_img_BL_img_", "_BL_mask_BL_img_", image_path)
            if not os.path.exists(label_path):
                continue
            raw_paths.append(image_path)
            label_paths.append(label_path)

    if timepoint in ("followup", "both"):
        for image_path in natsorted(glob(os.path.join(inputs_dir, "*_FU_img_FU_img_*.nii.gz"))):
            fname = re.sub(r"_FU_img_FU_img_", "_FU_mask_FU_img_", os.path.basename(image_path))
            label_path = os.path.join(targets_dir, fname)
            if not os.path.exists(label_path):
                continue
            raw_paths.append(image_path)
            label_paths.append(label_path)

    if len(raw_paths) == 0 or len(raw_paths) != len(label_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return raw_paths, label_paths


def get_longitudinal_ct_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    timepoint: Literal["baseline", "followup", "both"] = "both",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Longitudinal-CT dataset for lesion segmentation in whole-body CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        timepoint: The choice of timepoint. One of 'baseline', 'followup' or 'both' (default).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_longitudinal_ct_paths(path, timepoint, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_longitudinal_ct_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    timepoint: Literal["baseline", "followup", "both"] = "both",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Longitudinal-CT dataloader for lesion segmentation in whole-body CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        timepoint: The choice of timepoint. One of 'baseline', 'followup' or 'both' (default).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_longitudinal_ct_dataset(path, patch_shape, timepoint, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
