"""The CTPelvic1K dataset contains annotations for pelvic bones in CT scans.

This module provides the two subsets of the collection that are distributed with their own scans:
'clinic' with 103 annotated scans and 'clinic_metal' with 14 annotated scans of patients with metal
implants. The label ids are 1: sacrum, 2: hip (left), 3: hip (right), 4: lumbar vertebra.
See also `CLASS_IDS`.

NOTE: 'clinic_metal' ships 75 scans but only 14 of them are annotated, so only those are returned. Its
scans carry a 'dataset7_' prefix that its annotations do not, unlike the consistently named 'clinic'.

NOTE: The label ids of the two hips are assigned by the position of the annotation in the scan, which is
the reverse of the order that is usually quoted for this dataset. Every scan of the release is stored in
LPS orientation, and the annotation with id 2 lies on the left of the patient in all of them.

NOTE: The remaining subsets of the collection hold annotations for scans of other public datasets, such
as the colon task of the Medical Segmentation Decathlon and KiTS, and are not provided here.

The dataset is located at https://doi.org/10.5281/zenodo.4588403.
This dataset is from the publication https://doi.org/10.1007/s11548-021-02363-8.
Please cite it if you use this dataset in your research.
"""

import os
import re
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/4588403/files/{filename}?download=1"

SUBSETS = {
    "clinic": {"images": "CTPelvic1K_dataset6_data.tar.gz", "labels": "CTPelvic1K_dataset6_Anonymized_mask.tar.gz"},
    "clinic_metal": {"images": "CTPelvic1K_dataset7_data.tar.gz", "labels": "CTPelvic1K_dataset7_mask.tar.gz"},
}

CHECKSUMS = {
    "clinic": {
        "images": "9b8f4747d256483062aa937eb990890e59954d8be74966edd982a6dcf3f4394e",
        "labels": "382b0780bcbf68536631b14e9d2a42a0bccb6fc3c30c9598078ddfa0e574bf95",
    },
    "clinic_metal": {
        "images": "9b71fdf37c9bbb7bf99b95ea577f4bac24f9c393644e44287f59aad0b52a9cbc",
        "labels": "4b85e4eb7300e77ce094329933d032a88183f4f0ec65cde95e4162074bb7cca3",
    },
}

CLASS_NAMES = ["sacrum", "hip_left", "hip_right", "lumbar_vertebra"]
"""The pelvic bones of the CTPelvic1K dataset. The label id of a bone is its 1-based index."""

CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
"""Mapping from the bone name to its label id."""

LABEL_SUFFIX = "_mask_4label.nii.gz"

IMAGE_SUFFIX = "_data.nii.gz"


def _case_id(filename, suffix):
    """Strip the suffix and the subset prefix that the scans carry but their annotations do not."""
    return re.sub(r"^dataset\d+_", "", filename[:-len(suffix)])


def get_ctpelvic1k_data(
    path: Union[os.PathLike, str], subset: Literal["clinic", "clinic_metal"] = "clinic", download: bool = False
) -> str:
    """Download the CTPelvic1K dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        subset: The choice of subset. Either 'clinic' with 103 scans or 'clinic_metal' with 14 scans.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    if subset not in SUBSETS:
        raise ValueError(f"'{subset}' is not a valid subset. Choose from {list(SUBSETS.keys())}.")

    data_dir = os.path.join(path, subset)
    if os.path.exists(data_dir) and glob(os.path.join(data_dir, "**", f"*{LABEL_SUFFIX}"), recursive=True):
        return data_dir

    os.makedirs(data_dir, exist_ok=True)
    for key in ("images", "labels"):
        filename = SUBSETS[subset][key]
        tar_path = os.path.join(path, filename)
        util.download_source(
            path=tar_path, url=URL.format(filename=filename), download=download, checksum=CHECKSUMS[subset][key]
        )
        util.unzip_tarfile(tar_path=tar_path, dst=data_dir, remove=False)

    return data_dir


def get_ctpelvic1k_paths(
    path: Union[os.PathLike, str],
    subset: Literal["clinic", "clinic_metal"] = "clinic",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CTPelvic1K data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        subset: The choice of subset. Either 'clinic' with 103 scans or 'clinic_metal' with 14 scans.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_ctpelvic1k_data(path, subset, download)

    label_paths = natsorted(glob(os.path.join(data_dir, "**", f"*{LABEL_SUFFIX}"), recursive=True))
    image_paths = {
        _case_id(os.path.basename(p), IMAGE_SUFFIX): p
        for p in glob(os.path.join(data_dir, "**", f"*{IMAGE_SUFFIX}"), recursive=True)
    }

    # Only a part of the scans of 'clinic_metal' is annotated, so the annotations drive the pairing.
    raw_paths, valid_label_paths = [], []
    for label_path in label_paths:
        image_path = image_paths.get(_case_id(os.path.basename(label_path), LABEL_SUFFIX))
        if image_path is not None:
            raw_paths.append(image_path)
            valid_label_paths.append(label_path)

    assert len(raw_paths) == len(valid_label_paths) and len(raw_paths) > 0

    return raw_paths, valid_label_paths


def get_ctpelvic1k_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    subset: Literal["clinic", "clinic_metal"] = "clinic",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CTPelvic1K dataset for pelvic bone segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        subset: The choice of subset. Either 'clinic' with 103 scans or 'clinic_metal' with 14 scans.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_ctpelvic1k_paths(path, subset, download)

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


def get_ctpelvic1k_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    subset: Literal["clinic", "clinic_metal"] = "clinic",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CTPelvic1K dataloader for pelvic bone segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        subset: The choice of subset. Either 'clinic' with 103 scans or 'clinic_metal' with 14 scans.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ctpelvic1k_dataset(path, patch_shape, subset, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
