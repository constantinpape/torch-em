"""The Tear Meniscus dataset contains annotations for pixel-level segmentation of the lower tear meniscus
in external-eye colour and infrared images.

The dataset comprises 1693 colour images and 1739 infrared images, collected from five clinical centres
across China. The initial annotations were produced with a human-computer-interactive approach and were
subsequently reviewed and corrected by a senior ophthalmologist, so the ground truth is clinician-reviewed.
The shipped label maps are binary (0 = background, 1 = tear meniscus); some of them are stored as
3-channel (RGB) images with all channels holding the same grayscale value, which this module reduces to
a single-channel binary mask. The raw images also mix formats (RGBA, RGB and grayscale across the
different centres), which this module normalizes to 3-channel RGB.

The dataset is located at https://doi.org/10.6084/m9.figshare.28650536.v2 (CC BY 4.0).

This dataset is from the publication https://doi.org/10.1038/s41597-025-06460-0.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/56406179"
CHECKSUM = "b4614214e69098b09160f4713c5e0f11fcd6a7a158a5217ca9cd15ba2687f170"

LABEL_IDS = {"background": 0, "tear_meniscus": 1}

MODALITIES = {"colour": "Colour", "infrared": "Infrared"}


def get_tear_meniscus_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Tear Meniscus dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Open DataSet2")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "Open DataSet.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _preprocess_images(raw_paths: List[str]) -> List[str]:
    """Normalize the (RGBA / RGB / grayscale) raw images to 3-channel RGB."""
    preprocessed_paths = []
    for raw_path in tqdm(raw_paths, desc="Preprocessing images"):
        parent = Path(raw_path).parent.parent
        preprocessed_dir = os.path.join(parent, "Original_rgb")
        os.makedirs(preprocessed_dir, exist_ok=True)

        preprocessed_path = os.path.join(preprocessed_dir, f"{Path(raw_path).stem}.tif")
        preprocessed_paths.append(preprocessed_path)
        if os.path.exists(preprocessed_path):
            continue

        raw = imageio.imread(raw_path)
        if raw.ndim == 2:
            raw = np.repeat(raw[..., None], 3, axis=-1)
        else:
            raw = raw[..., :3]
        imageio.imwrite(preprocessed_path, raw, compression="zlib")

    return preprocessed_paths


def _preprocess_labels(raw_label_paths: List[str]) -> List[str]:
    """Reduce the (partially 3-channel) label images to single-channel binary masks."""
    preprocessed_paths = []
    for label_path in tqdm(raw_label_paths, desc="Preprocessing labels"):
        parent = Path(label_path).parent.parent
        preprocessed_dir = os.path.join(parent, "Label_binary")
        os.makedirs(preprocessed_dir, exist_ok=True)

        preprocessed_path = os.path.join(preprocessed_dir, f"{Path(label_path).stem}.tif")
        preprocessed_paths.append(preprocessed_path)
        if os.path.exists(preprocessed_path):
            continue

        label = imageio.imread(label_path)
        if label.ndim == 3:
            label = label[..., 0]
        label = (label > 127).astype("uint8")
        imageio.imwrite(preprocessed_path, label, compression="zlib")

    return preprocessed_paths


def get_tear_meniscus_paths(
    path: Union[os.PathLike, str],
    modality: Literal["colour", "infrared"] = "colour",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Tear Meniscus data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The imaging modality. Either 'colour' or 'infrared'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_tear_meniscus_data(path, download)

    if modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Please choose one of {list(MODALITIES)}.")
    prefix = MODALITIES[modality]

    raw_image_paths = natsorted(glob(os.path.join(data_dir, f"{prefix}*", "Original", "*.PNG")))
    raw_label_paths = [p.replace(f"{os.sep}Original{os.sep}", f"{os.sep}Label{os.sep}") for p in raw_image_paths]

    assert len(raw_image_paths) > 0, f"Could not find any images in '{data_dir}'."
    for label_path in raw_label_paths:
        assert os.path.exists(label_path), label_path

    image_paths = _preprocess_images(raw_image_paths)
    label_paths = _preprocess_labels(raw_label_paths)

    return image_paths, label_paths


def get_tear_meniscus_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    modality: Literal["colour", "infrared"] = "colour",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Tear Meniscus dataset for segmentation of the tear meniscus.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The imaging modality. Either 'colour' or 'infrared'.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_tear_meniscus_paths(path, modality, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_tear_meniscus_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    modality: Literal["colour", "infrared"] = "colour",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Tear Meniscus dataloader for segmentation of the tear meniscus.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The imaging modality. Either 'colour' or 'infrared'.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_tear_meniscus_dataset(path, patch_shape, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
