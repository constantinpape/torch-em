"""HVM (Hepatic Vessel Map) is a dual-center dataset for the segmentation of hepatic veins, portal
veins (to third-order branches) and liver tumors in contrast-enhanced abdominal CT.

The dataset comprises 282 patients: 170 scans from Center 1 (The University of Hong Kong-Shenzhen
Hospital, of which 106 have hepatic and portal vein annotations) and 176 scans from Center 2
(Peking University Shenzhen Hospital, of which 174 have hepatic and portal vein annotations). Liver
tumor annotations (96 cases) are only provided for a subset of Center 1 (the archive also ships an
identical copy of these files under Center 2, which does not correspond to any Center 2 scan and is
therefore not used by this loader).

NOTE: The raw scans and the annotations in each folder are not matched by their filename id (e.g.
'Image/1.nii.gz' does not necessarily correspond to 'Annotation_Hepatic veins/001.nii.gz'). They are
instead matched here by comparing the NIfTI header geometry (shape, voxel spacing and origin), which
uniquely identifies the corresponding scan for (almost) every annotation.

This dataset is NOT the same as the 'HVA-CT' dataset (see `hva_ct.py`), which re-annotates the 61
scans of the Medical Segmentation Decathlon hepatic vessel task.

The dataset is located at https://doi.org/10.5281/zenodo.19885789 and is distributed under the
CC BY 4.0 license. The dataset is from the publication https://doi.org/10.1038/s41597-026-07550-3.
Please cite it if you use this dataset for your research.
"""

import os
import warnings
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import nibabel as nib

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/19885789/files/HVM%20Dataset.zip/content"
CHECKSUM = "7fe2a00bc8a40658d45bedd27cd70df22e1a58a92af20037c22b05065407a6b8"

CENTERS = ["1", "2"]

ANNOTATIONS = {
    "hepatic_veins": "Annotation_Hepatic veins",
    "portal_veins": "Annotation_Portal veins",
    "liver_tumor": "Annotation_Liver tumors",
}


def get_hvm_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the HVM dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "HVM Dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "HVM_Dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _header_signature(path):
    header = nib.load(path).header
    shape = header.get_data_shape()
    zooms = tuple(header.get_zooms())
    origin = tuple(header.get_best_affine()[:3, 3])
    return shape, zooms, origin


def _match_images_to_annotations(image_paths, annotation_paths):
    image_sigs = {p: _header_signature(p) for p in image_paths}

    matched_images, matched_annotations = [], []
    for annotation_path in annotation_paths:
        annotation_sig = _header_signature(annotation_path)
        candidates = [p for p, sig in image_sigs.items() if sig == annotation_sig]
        if not candidates:
            warnings.warn(f"Could not find a matching scan for the annotation at '{annotation_path}'. Skipping it.")
            continue
        matched_images.append(natsorted(candidates)[0])
        matched_annotations.append(annotation_path)

    return matched_images, matched_annotations


def get_hvm_paths(
    path: Union[os.PathLike, str],
    annotation: Literal["hepatic_veins", "portal_veins", "liver_tumor"] = "hepatic_veins",
    center: Literal["1", "2", "both"] = "both",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the HVM data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        annotation: The choice of annotations. One of 'hepatic_veins', 'portal_veins' or 'liver_tumor'.
        center: The choice of source center. One of '1', '2' or 'both'. Ignored (forced to '1') when
            `annotation` is 'liver_tumor', as the tumor annotations shipped under 'Center 2' are an
            identical copy of the 'Center 1' ones and do not correspond to any 'Center 2' scan.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if annotation not in ANNOTATIONS:
        raise ValueError(f"'{annotation}' is not a valid annotation. Choose from {list(ANNOTATIONS.keys())}.")
    if center not in CENTERS + ["both"]:
        raise ValueError(f"'{center}' is not a valid center. Choose from {CENTERS + ['both']}.")

    data_dir = get_hvm_data(path=path, download=download)

    centers = ["1"] if annotation == "liver_tumor" else (CENTERS if center == "both" else [center])

    image_paths, gt_paths = [], []
    for c in centers:
        center_dir = os.path.join(data_dir, f"Center {c}")
        image_dir = os.path.join(center_dir, "Image")
        annotation_dir = os.path.join(center_dir, ANNOTATIONS[annotation])

        this_images = natsorted(glob(os.path.join(image_dir, "*.nii.gz")))
        this_annotations = natsorted(glob(os.path.join(annotation_dir, "*.nii.gz")))

        matched_images, matched_annotations = _match_images_to_annotations(this_images, this_annotations)
        image_paths.extend(matched_images)
        gt_paths.extend(matched_annotations)

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0

    return image_paths, gt_paths


def get_hvm_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    annotation: Literal["hepatic_veins", "portal_veins", "liver_tumor"] = "hepatic_veins",
    center: Literal["1", "2", "both"] = "both",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HVM dataset for hepatic vein, portal vein and liver tumor segmentation in abdominal CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        annotation: The choice of annotations. One of 'hepatic_veins', 'portal_veins' or 'liver_tumor'.
        center: The choice of source center. One of '1', '2' or 'both'. Ignored (forced to '1') when
            `annotation` is 'liver_tumor'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_hvm_paths(path, annotation, center, download)

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


def get_hvm_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    annotation: Literal["hepatic_veins", "portal_veins", "liver_tumor"] = "hepatic_veins",
    center: Literal["1", "2", "both"] = "both",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the HVM dataloader for hepatic vein, portal vein and liver tumor segmentation in abdominal CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        annotation: The choice of annotations. One of 'hepatic_veins', 'portal_veins' or 'liver_tumor'.
        center: The choice of source center. One of '1', '2' or 'both'. Ignored (forced to '1') when
            `annotation` is 'liver_tumor'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hvm_dataset(path, patch_shape, annotation, center, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
