"""The LSS MRI AISSLab dataset contains annotations for vertebra, intervertebral disc, sacrum
and posterior structure segmentation in sagittal lumbar spine MRI.

The dataset consists of 500 patients with sagittal lumbar spine MRI, approved by the IRB of
Firat University and clinically validated by neurosurgeons. For each patient, the middle sagittal
slice is provided as a PNG image together with a pixel-level segmentation mask that was AI-initialized
and then manually refined / verified by the neurosurgeons. The mask labels are (pixel value: label):
0 = anterior background, 50 = vertebra, 100 = intervertebral disc, 150 = sacrum,
200 = posterior A, 255 = posterior B.

NOTE: The full release also ships the raw sagittal DICOM series per patient and PNG / XML foraminal
stenosis bounding box annotations on all slices. This module only downloads and exposes the
middle-slice images and their pixel-level segmentation masks, as the DICOM series and the stenosis
bounding boxes are not relevant for segmentation.

The dataset is located at https://data.mendeley.com/datasets/rgb77xm3jf/4 (CC BY 4.0).
This dataset is from the publication https://doi.org/10.1038/s41597-026-07138-x.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://data.mendeley.com/public-files/datasets/rgb77xm3jf/files/6d9a0116-925d-4111-acb0-1e679f7dfd71/file_downloaded"  # noqa
CHECKSUM = "592a294f93d575a16bccc2681c793eb1cfc6679fa2746ac50cbc8970f806b4b1"

# The pixel values used in the 'Segmentation/Masks/*M.png' files, mapped to contiguous label ids.
LABEL_IDS = {0: 0, 50: 1, 100: 2, 150: 3, 200: 4, 255: 5}


def get_lss_mri_aisslab_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the LSS MRI AISSLab dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the 'Segmentation' data is stored.
    """
    data_dir = os.path.join(path, "Segmentation")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "LSS_MRI_AISSLab.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)

    import zipfile
    with zipfile.ZipFile(zip_path) as f:
        members = [
            m for m in f.namelist()
            if m.startswith("Segmentation/Middle_Slice/") or m.startswith("Segmentation/Masks/")
        ]
        f.extractall(path, members=members)
    os.remove(zip_path)

    return data_dir


def _preprocess_masks(data_dir):
    mask_paths = natsorted(glob(os.path.join(data_dir, "Masks", "*M.png")))
    neu_dir = os.path.join(data_dir, "preprocessed_masks")
    os.makedirs(neu_dir, exist_ok=True)

    neu_mask_paths = []
    for mask_path in mask_paths:
        neu_path = os.path.join(neu_dir, os.path.basename(mask_path))
        if not os.path.exists(neu_path):
            mask = imageio.imread(mask_path)
            neu_mask = np.zeros_like(mask, dtype="uint8")
            for value, label_id in LABEL_IDS.items():
                neu_mask[mask == value] = label_id
            imageio.imwrite(neu_path, neu_mask)
        neu_mask_paths.append(neu_path)

    return neu_mask_paths


def get_lss_mri_aisslab_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the LSS MRI AISSLab data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_lss_mri_aisslab_data(path, download)

    image_paths = natsorted(glob(os.path.join(data_dir, "Middle_Slice", "*.png")))
    label_paths = _preprocess_masks(data_dir)

    assert len(image_paths) > 0 and len(image_paths) == len(label_paths), (
        f"Expected the same number of images and masks, got {len(image_paths)} and {len(label_paths)}."
    )

    return image_paths, label_paths


def get_lss_mri_aisslab_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LSS MRI AISSLab dataset for lumbar spine segmentation in sagittal MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_lss_mri_aisslab_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_lss_mri_aisslab_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LSS MRI AISSLab dataloader for lumbar spine segmentation in sagittal MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lss_mri_aisslab_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
