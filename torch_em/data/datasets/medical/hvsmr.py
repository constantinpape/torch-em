"""The HVSMR-2.0 dataset contains annotations for whole-heart segmentation in 3D cardiovascular MRI
of patients with congenital heart disease.

This module implements the HVSMR-2.0 release (2024), not the older HVSMR 2016 challenge data
(https://segchd.csail.mit.edu), which labels only the blood pool and the ventricular myocardium.
HVSMR-2.0 consists of 60 cardiovascular MR scans acquired at Boston Children's Hospital, with manual
segmentations of the four cardiac chambers and the four great vessels. It does not contain a myocardium
label. The label ids are described in `LABEL_IDS`: 1 = left ventricle, 2 = right ventricle, 3 = left atrium,
4 = right atrium, 5 = aorta, 6 = pulmonary artery, 7 = superior vena cava, 8 = inferior vena cava.
Some chambers are missing in single-ventricle and common-atrium patients, which is anatomy and not an
annotation error.

The release is distributed in three variants, which are selected with the 'variant' argument:
- 'orig': the images cropped at the chin, with the original spacing and without normalization.
- 'cropped': the images cropped around the heart (the default, and the smallest download).
- 'cropped_norm': the heart-cropped images with normalized intensities.
Each variant also ships 'pat<n>_*_seg_endpoints.nii.gz' files, which delineate the optional extents of the
great vessels for a fairer evaluation. They are not exposed as labels by this module.

The data is located at https://doi.org/10.6084/m9.figshare.c.7074755 and is licensed under CC BY 4.0.

This dataset is from the publication https://doi.org/10.1038/s41597-024-03469-9.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "orig": "https://ndownloader.figshare.com/files/44561774",
    "cropped": "https://ndownloader.figshare.com/files/44561792",
    "cropped_norm": "https://ndownloader.figshare.com/files/44561783",
}

CHECKSUMS = {
    "orig": "1fb39b1a9ad040f5860eb39cdb6ee673d998e7c5a3b1b3fa06ae5d81d8fa3bb4",
    "cropped": "62338737c1cb8cf690f4d2dad770caae9590c4e757b2ecc44ce6d1f33aa2d005",
    "cropped_norm": "c1c101117e195b143ec4b66a5a35501766efe907a62db2655ef6024fa655f0fc",
}

LABEL_IDS = {
    "background": 0,
    "LV": 1,
    "RV": 2,
    "LA": 3,
    "RA": 4,
    "AO": 5,
    "PA": 6,
    "SVC": 7,
    "IVC": 8,
}

VARIANTS = list(URLS.keys())


def get_hvsmr_data(
    path: Union[os.PathLike, str],
    variant: Literal["orig", "cropped", "cropped_norm"] = "cropped",
    download: bool = False,
) -> str:
    """Download the HVSMR-2.0 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        variant: The choice of image variant. Either 'orig', 'cropped' or 'cropped_norm'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    if variant not in VARIANTS:
        raise ValueError(f"'{variant}' is not a valid variant. Please choose one of {VARIANTS}.")

    data_dir = os.path.join(path, variant)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"{variant}.zip")
    util.download_source(path=zip_path, url=URLS[variant], download=download, checksum=CHECKSUMS[variant])
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_hvsmr_paths(
    path: Union[os.PathLike, str],
    variant: Literal["orig", "cropped", "cropped_norm"] = "cropped",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the HVSMR-2.0 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        variant: The choice of image variant. Either 'orig', 'cropped' or 'cropped_norm'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_hvsmr_data(path, variant, download)

    # The 'cropped_norm' images are named after the 'cropped' variant, but with a '_norm' suffix.
    image_suffix = "cropped_norm" if variant == "cropped_norm" else variant
    label_suffix = "cropped" if variant == "cropped_norm" else variant

    label_paths = natsorted(glob(os.path.join(data_dir, f"pat*_{label_suffix}_seg.nii.gz")))
    raw_paths = [p.replace(f"_{label_suffix}_seg.nii.gz", f"_{image_suffix}.nii.gz") for p in label_paths]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in raw_paths)

    return raw_paths, label_paths


def get_hvsmr_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    variant: Literal["orig", "cropped", "cropped_norm"] = "cropped",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HVSMR-2.0 dataset for whole-heart segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        variant: The choice of image variant. Either 'orig', 'cropped' or 'cropped_norm'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_hvsmr_paths(path, variant, download)

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


def get_hvsmr_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    variant: Literal["orig", "cropped", "cropped_norm"] = "cropped",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the HVSMR-2.0 dataloader for whole-heart segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        variant: The choice of image variant. Either 'orig', 'cropped' or 'cropped_norm'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hvsmr_dataset(path, patch_shape, variant, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
