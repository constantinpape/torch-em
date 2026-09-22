"""This dataset contains pixel-level segmented longitudinal OCT scans of idiopathic
full-thickness macular hole (iFTMH) surgery outcomes, collected at one preoperative
('baseline') and six postoperative time points ('2weeks', '3months', '6months',
'12months', '24months', '48months').

The dataset contains 2,591 fovea-centered horizontal and vertical B-scans from 493
patients, with expert-validated pixel-level segmentation masks for 12 retinal
structures and pathologies, both anatomical (e.g. external limiting membrane,
ellipsoid zone, retinal pigment epithelium) and pathological (e.g. macular hole,
cysts, epiretinal membrane, subretinal fluid).

NOTE: This is distinct from the OIMHS dataset (see 'torch_em/data/datasets/medical/
oimhs.py'), which is a different, non-longitudinal macular hole OCT dataset.

The dataset is located at https://doi.org/10.6084/m9.figshare.32605218 and is
licensed under CC-BY-4.0.

This dataset is from the publication https://doi.org/10.1038/s41597-026-08154-7.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/65374524"
CHECKSUM = "e9956c1587123e94342cc74dd9d2a50c9c833ffe52457688862ece8b849a4c97"

TIMEPOINTS = ["baseline", "2weeks", "3months", "6months", "12months", "24months", "48months"]


def get_longitudinal_mh_oct_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the longitudinal macular hole OCT data.

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


def get_longitudinal_mh_oct_paths(
    path: Union[os.PathLike, str],
    timepoint: Optional[Literal["baseline", "2weeks", "3months", "6months", "12months", "24months", "48months"]] = None,  # noqa
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the longitudinal macular hole OCT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        timepoint: The choice of a specific postoperative (or 'baseline') timepoint. By default, loads all
            timepoints.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_longitudinal_mh_oct_data(path, download)

    timepoints = TIMEPOINTS if timepoint is None else [timepoint]

    pp_dir = os.path.join(data_dir, "preprocessed_images")
    os.makedirs(pp_dir, exist_ok=True)

    image_paths, gt_paths = [], []
    for this_timepoint in timepoints:
        this_gt_paths = natsorted(
            glob(os.path.join(data_dir, this_timepoint, f"{this_timepoint}_Masks", "*.png"))
        )
        for gt_path in this_gt_paths:
            org_image_path = os.path.join(
                data_dir, this_timepoint, f"{this_timepoint}_OCT", f"{Path(gt_path).stem}.tiff"
            )
            if not os.path.exists(org_image_path):
                continue

            # The raw B-scans are RGBA tiffs, but 'ImageCollectionDataset' expects RGB inputs.
            # The alpha channel is dropped once here and the result cached as a '.tif' file.
            image_path = os.path.join(pp_dir, f"{this_timepoint}_{Path(gt_path).stem}.tif")
            if not os.path.exists(image_path):
                image = imageio.imread(org_image_path)
                if image.ndim == 3 and image.shape[-1] == 4:
                    image = image[..., :3]
                imageio.imwrite(image_path, image, compression="zlib")

            image_paths.append(image_path)
            gt_paths.append(gt_path)

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0, (
        "No image-mask pairs were found. The expected per-timepoint '<timepoint>_OCT' / '<timepoint>_Masks' "
        "folder layout may not match the actual structure of the downloaded data. Please inspect the data at "
        f"'{data_dir}'."
    )

    return image_paths, gt_paths


def get_longitudinal_mh_oct_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    timepoint: Optional[Literal["baseline", "2weeks", "3months", "6months", "12months", "24months", "48months"]] = None,  # noqa
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the longitudinal macular hole OCT dataset for segmentation of 12 retinal structures and pathologies.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        timepoint: The choice of a specific postoperative (or 'baseline') timepoint. By default, loads all
            timepoints.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_longitudinal_mh_oct_paths(path, timepoint, download)

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


def get_longitudinal_mh_oct_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    timepoint: Optional[Literal["baseline", "2weeks", "3months", "6months", "12months", "24months", "48months"]] = None,  # noqa
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the longitudinal macular hole OCT dataloader for segmentation of 12 retinal structures and pathologies.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        timepoint: The choice of a specific postoperative (or 'baseline') timepoint. By default, loads all
            timepoints.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_longitudinal_mh_oct_dataset(path, patch_shape, timepoint, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
