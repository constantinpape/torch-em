"""The Couinaud dataset contains annotations for the eight Couinaud liver segments in CT scans.

The annotations were made on the CT scans of the Medical Segmentation Decathlon task 8 (hepatic vessel),
so the images are downloaded from there. Two annotation sets are provided: 'couinaud' with the eight
Couinaud segments for 193 scans and 'liver' with a binary liver mask for 443 scans. The label ids of the
'couinaud' annotations are 1 to 8 for the Couinaud segments I to VIII. See also `CLASS_IDS`.

The dataset is located at https://github.com/GLCUnet/dataset and is distributed under the MIT license.
This dataset is from the publication https://doi.org/10.1007/978-3-030-32692-0_32.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .msd import get_msd_data
from .. import util


URLS = {
    "couinaud": "https://raw.githubusercontent.com/GLCUnet/dataset/master/couinaud_annotation.zip",
    "liver": "https://raw.githubusercontent.com/GLCUnet/dataset/master/liver_annotation.zip",
}

CHECKSUMS = {
    "couinaud": "fb2fc7809a7982267adc2785dddd5af3770070fea230e2f01142fe799a29f0cf",
    "liver": "e14148cec317829a1d4068e82f60cfecea6df570decd7c06b474f172aff382a1",
}

CLASS_NAMES = [
    "segment_i", "segment_ii", "segment_iii", "segment_iv",
    "segment_v", "segment_vi", "segment_vii", "segment_viii",
]
"""The Couinaud liver segments. The label id of a segment is its 1-based index."""

CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
"""Mapping from the Couinaud segment name to its label id."""


def _get_image_paths(msd_dir):
    image_paths = {}
    for split in ["imagesTr", "imagesTs"]:
        for path in glob(os.path.join(msd_dir, "Task08_HepaticVessel", split, "*.nii.gz")):
            fname = os.path.basename(path)
            # The MSD archives carry macOS resource fork files next to the actual volumes.
            if fname.startswith("._"):
                continue
            image_paths[fname] = path
    return image_paths


def get_couinaud_data(
    path: Union[os.PathLike, str], annotation: Literal["couinaud", "liver"] = "couinaud", download: bool = False
) -> Tuple[str, str]:
    """Download the Couinaud dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        annotation: The choice of annotations. Either 'couinaud' for the eight Couinaud segments
            or 'liver' for a binary liver mask.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the annotations are downloaded.
        Filepath where the images are downloaded.
    """
    if annotation not in URLS:
        raise ValueError(f"'{annotation}' is not a valid annotation. Choose from {list(URLS.keys())}.")

    label_dir = os.path.join(path, f"{annotation}_annotation")
    if not os.path.exists(label_dir):
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, f"{annotation}_annotation.zip")
        util.download_source(
            path=zip_path, url=URLS[annotation], download=download, checksum=CHECKSUMS[annotation]
        )
        util.unzip(zip_path=zip_path, dst=path, remove=False)

    # The images are the hepatic vessel scans of the Medical Segmentation Decathlon.
    msd_dir = get_msd_data(path=path, task_name="hepaticvessel", download=download)

    return label_dir, msd_dir


def get_couinaud_paths(
    path: Union[os.PathLike, str],
    annotation: Literal["couinaud", "liver"] = "couinaud",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Couinaud data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        annotation: The choice of annotations. Either 'couinaud' for the eight Couinaud segments
            or 'liver' for a binary liver mask.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    label_dir, msd_dir = get_couinaud_data(path, annotation, download)

    image_paths = _get_image_paths(msd_dir)
    label_paths = natsorted(glob(os.path.join(label_dir, "*.nii.gz")))

    raw_paths, valid_label_paths = [], []
    for label_path in label_paths:
        image_path = image_paths.get(os.path.basename(label_path))
        if image_path is not None:
            raw_paths.append(image_path)
            valid_label_paths.append(label_path)

    assert len(raw_paths) == len(valid_label_paths) and len(raw_paths) > 0

    return raw_paths, valid_label_paths


def get_couinaud_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    annotation: Literal["couinaud", "liver"] = "couinaud",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Couinaud dataset for liver segment segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        annotation: The choice of annotations. Either 'couinaud' for the eight Couinaud segments
            or 'liver' for a binary liver mask.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_couinaud_paths(path, annotation, download)

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


def get_couinaud_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    annotation: Literal["couinaud", "liver"] = "couinaud",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Couinaud dataloader for liver segment segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        annotation: The choice of annotations. Either 'couinaud' for the eight Couinaud segments
            or 'liver' for a binary liver mask.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_couinaud_dataset(path, patch_shape, annotation, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
