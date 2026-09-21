"""The OpenSwissHCC dataset contains annotations for liver and hepatocellular carcinoma (HCC)
lesion segmentation in multiparametric, multiphasic liver MRI.

The dataset is located at https://doi.org/10.5281/zenodo.21992461. It comprises 132 DCE-MRI
examinations (63 HCC-positive, 69 HCC-negative patients) with up to five manually annotated
lesions per subject (140 lesions in total, 97 confirmed HCC), each delineated on every sequence
and phase where visible. A subset of 16 subjects additionally carries manual whole-liver masks.

The dataset also ships automatically-generated (nnU-Net) whole-liver pseudo-labels for all
subjects; these are NOT exposed by this module, which only provides the manual lesion and manual
liver annotations.

The dataset is from the publication https://doi.org/10.3390/tomography12090133.
Please cite it if you use this dataset for your research.
"""

import os
import re
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "sub-001-sub-044": "https://zenodo.org/records/21992461/files/sub-001-sub-044.zip",
    "sub-044-sub-088": "https://zenodo.org/records/21992461/files/sub-044-sub-088.zip",
    "sub-088-sub-132": "https://zenodo.org/records/21992461/files/sub-088-sub-132.zip",
    "derivatives": "https://zenodo.org/records/21992461/files/derivatives.zip",
}

CHECKSUMS = {
    "derivatives": "6734f3b9b484c4d04f2f3258a56ee8552f56894ac25f77fe47f72e85ed0c0308",
}
"""NOTE: The three large raw MRI archives ('sub-001-sub-044', 'sub-044-sub-088', 'sub-088-sub-132')
are not checksummed here (`get_openswisshcc_data` passes `checksum=None` for them), since their
size (4-5GB each) makes ad-hoc verification impractical; only the small 'derivatives' archive is
checksummed."""

LESION_SUFFIX_PATTERN = re.compile(r"-L\d+_seg\.nii\.gz$")
LIVER_SUFFIX_PATTERN = re.compile(r"-liver_seg(-annotator)?\.nii\.gz$")


def get_openswisshcc_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the OpenSwissHCC dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    raw_dir = os.path.join(path, "raw")
    derivatives_dir = os.path.join(path, "derivatives")

    os.makedirs(path, exist_ok=True)

    for name in ("sub-001-sub-044", "sub-044-sub-088", "sub-088-sub-132"):
        if os.path.exists(os.path.join(raw_dir, name)):
            continue
        zip_path = os.path.join(path, f"{name}.zip")
        util.download_source(path=zip_path, url=URLS[name], download=download, checksum=None)
        util.unzip(zip_path=zip_path, dst=raw_dir, remove=True)

    if not os.path.exists(derivatives_dir):
        zip_path = os.path.join(path, "derivatives.zip")
        util.download_source(
            path=zip_path, url=URLS["derivatives"], download=download, checksum=CHECKSUMS["derivatives"]
        )
        util.unzip(zip_path=zip_path, dst=path, remove=True)

    return path


def _build_raw_index(raw_dir):
    index = {}
    for p in glob(os.path.join(raw_dir, "*", "sub-*", "*", "*.nii.gz")):
        parts = Path(p).parts
        rel_key = os.path.join(*parts[-3:])  # sub-XXX/<subdir>/<basename>.nii.gz
        index[rel_key] = p
    return index


def get_openswisshcc_paths(
    path: Union[os.PathLike, str],
    label_choice: Literal["lesion", "liver"] = "lesion",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the OpenSwissHCC data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        label_choice: The choice of manual annotation. Either 'lesion' (up to five manually
            annotated lesions per subject, across all subjects and sequences/phases where the
            lesion is visible) or 'liver' (manual whole-liver masks, available for a subset of
            16 subjects).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if label_choice not in ("lesion", "liver"):
        raise ValueError(f"'{label_choice}' is not a valid label choice. Please choose 'lesion' or 'liver'.")

    data_dir = get_openswisshcc_data(path, download)
    raw_dir = os.path.join(data_dir, "raw")
    raw_index = _build_raw_index(raw_dir)

    if label_choice == "lesion":
        derivatives_subdir = os.path.join(data_dir, "derivatives", "manual_lesion_annotations")
        suffix_pattern = LESION_SUFFIX_PATTERN
    else:
        derivatives_subdir = os.path.join(data_dir, "derivatives", "manual_liver_annotations")
        suffix_pattern = LIVER_SUFFIX_PATTERN

    mask_paths = natsorted(glob(os.path.join(derivatives_subdir, "sub-*", "*", "*.nii.gz")))

    image_paths, gt_paths = [], []
    for mask_path in mask_paths:
        parts = Path(mask_path).parts
        subject, subdir, basename = parts[-3], parts[-2], parts[-1]

        raw_basename = suffix_pattern.sub(".nii.gz", basename)
        if raw_basename == basename:  # the suffix pattern did not match, skip this file.
            continue

        rel_key = os.path.join(subject, subdir, raw_basename)
        raw_path = raw_index.get(rel_key)
        if raw_path is None:
            continue

        # A handful of masks in the source archive were annotated on a cropped sub-volume and so
        # do not share the raw volume's number of slices; skip pairs with a shape mismatch rather
        # than letting them fail deep in `SegmentationDataset.__init__`.
        import nibabel as nib
        if nib.load(raw_path).shape != nib.load(mask_path).shape:
            continue

        image_paths.append(raw_path)
        gt_paths.append(mask_path)

    return image_paths, gt_paths


def get_openswisshcc_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    label_choice: Literal["lesion", "liver"] = "lesion",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the OpenSwissHCC dataset for segmentation of liver / HCC lesions in multiphasic MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of manual annotation. Either 'lesion' or 'liver'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_openswisshcc_paths(path, label_choice, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_openswisshcc_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    label_choice: Literal["lesion", "liver"] = "lesion",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the OpenSwissHCC dataloader for segmentation of liver / HCC lesions in multiphasic MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of manual annotation. Either 'lesion' or 'liver'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_openswisshcc_dataset(path, patch_shape, label_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
