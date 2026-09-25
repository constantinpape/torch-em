"""The TRUSTED dataset contains annotations for kidney segmentation in paired 3d
transabdominal ultrasound and CT volumes.

The dataset contains 96 kidneys from 48 human patients, imaged with both transabdominal
3d ultrasound and CT. Two independent radiographers manually segmented each kidney, and
a STAPLE-fused consensus segmentation is provided in addition to the individual annotations.

This dataset is located at https://doi.org/10.6084/m9.figshare.27981050.v1 (CC BY 4.0).
This dataset is from the publication https://doi.org/10.1038/s41597-025-04467-1.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/51079133"
CHECKSUM = "2e63e560f4dcbccba920cd90e2add9738da01efa8e5f6834bd36c4768757b0b4"

# The estimated STAPLE-fused consensus masks contain tiny near-zero floating point noise
# (e.g. ~1e-14) alongside actual foreground values (~1.0) instead of clean binary values.
# We binarize the labels on the fly (per sampled patch) rather than rewriting the full
# 3d volumes to disk, since the volumes are large (up to ~300MB each, uncompressed GBs).


def _binarize_labels(labels):
    return (labels > 0.5).astype("float32")


def get_trusted_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the TRUSTED dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "TRUSTED_dataset_for_nsd")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "TRUSTED_dataset_for_nsd.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_trusted_paths(
    path: Union[os.PathLike, str],
    modality: Literal["us", "ct"] = "us",
    label_choice: Literal["gt_estimated", "annotator1", "annotator2"] = "gt_estimated",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the TRUSTED data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The choice of imaging modality, either 3d transabdominal ultrasound ('us') or CT ('ct').
        label_choice: The choice of segmentation annotation to use, either the STAPLE-fused
            consensus mask ('gt_estimated') or one of the two individual annotators
            ('annotator1' / 'annotator2').
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_trusted_data(path=path, download=download)

    if modality not in ["us", "ct"]:
        raise ValueError(f"'{modality}' is not a valid modality choice.")

    if label_choice not in ["gt_estimated", "annotator1", "annotator2"]:
        raise ValueError(f"'{label_choice}' is not a valid label choice.")

    suffix = modality.upper()
    image_dir = os.path.join(data_dir, f"{suffix}_DATA", f"{suffix}_images")
    mask_dir = os.path.join(data_dir, f"{suffix}_DATA", f"{suffix}_masks")

    all_image_paths = sorted(glob(os.path.join(image_dir, f"*_img{suffix}.nii.gz")))

    image_paths, gt_paths = [], []
    for image_path in all_image_paths:
        case_id = os.path.basename(image_path).replace(f"_img{suffix}.nii.gz", "")
        if label_choice == "gt_estimated":
            gt_path = os.path.join(mask_dir, f"GT_estimated_masks{suffix}", f"{case_id}_mask{suffix}.nii.gz")
        elif label_choice == "annotator1":
            annotator_id = case_id + "1" if modality == "us" else case_id + "_1"
            gt_path = os.path.join(mask_dir, "Annotator1", f"{annotator_id}_mask{suffix}.nii.gz")
        else:
            annotator_id = case_id + "2" if modality == "us" else case_id + "_2"
            gt_path = os.path.join(mask_dir, "Annotator2", f"{annotator_id}_mask{suffix}.nii.gz")

        # Not every case has annotations from both annotators, so we skip the ones that are missing.
        if os.path.exists(gt_path):
            image_paths.append(image_path)
            gt_paths.append(gt_path)

    if len(image_paths) == 0 or len(image_paths) != len(gt_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return image_paths, gt_paths


def get_trusted_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    modality: Literal["us", "ct"] = "us",
    label_choice: Literal["gt_estimated", "annotator1", "annotator2"] = "gt_estimated",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the TRUSTED dataset for kidney segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The choice of imaging modality, either 3d transabdominal ultrasound ('us') or CT ('ct').
        label_choice: The choice of segmentation annotation to use, either the STAPLE-fused
            consensus mask ('gt_estimated') or one of the two individual annotators
            ('annotator1' / 'annotator2').
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_trusted_paths(path, modality, label_choice, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    kwargs.setdefault("pre_label_transform", _binarize_labels)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        **kwargs
    )


def get_trusted_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    modality: Literal["us", "ct"] = "us",
    label_choice: Literal["gt_estimated", "annotator1", "annotator2"] = "gt_estimated",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the TRUSTED dataloader for kidney segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The choice of imaging modality, either 3d transabdominal ultrasound ('us') or CT ('ct').
        label_choice: The choice of segmentation annotation to use, either the STAPLE-fused
            consensus mask ('gt_estimated') or one of the two individual annotators
            ('annotator1' / 'annotator2').
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_trusted_dataset(path, patch_shape, modality, label_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
