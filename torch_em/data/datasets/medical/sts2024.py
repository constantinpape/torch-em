"""The STS2024 dataset contains annotations for instance-level tooth segmentation, labeled by
FDI tooth id, in panoramic dental X-rays (OPGs).

The dataset was curated for the 2nd Semi-supervised Teeth Segmentation (STS 2024) MICCAI challenge
(https://sts-challenge.github.io/miccai2024/index.html), which extends the STS 2023 challenge to
multi-instance, multi-class (FDI enumeration) instance-level tooth segmentation in both panoramic
X-rays and CBCT scans. The full challenge data comprises 2,380 OPGs and 330 CBCT volumes, but only
30 OPG cases and 30 CBCT cases carry full ground truth (the remaining images support the challenge's
semi-supervised learning setting and have no public labels). This module only covers the 30 fully
labeled OPG cases, released on Zenodo; the CBCT ground truth is not part of that release.

The original annotations are per-tooth polygons (in labelme format) with the FDI tooth id as the
label; this module rasterizes them into a per-pixel label map, where the pixel value is the FDI id
of the tooth (0 marks background).

The data is hosted on Zenodo at https://zenodo.org/records/17712688 and is distributed under the
CC BY 4.0 license.

The dataset is from the publication https://doi.org/10.1016/j.media.2026.103986.
Please cite it if you use this dataset for your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
from skimage.draw import polygon

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/17712688/files/Train-Labeled.zip?download=1"
CHECKSUM = "8d0e1491368f15c770f592d6c9284861629140e6bb550659257a3ff13b4b9761"


def get_sts2024_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the STS2024 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Train-Labeled")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "Train-Labeled.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _rasterize_annotations(shape, annotation_path):
    with open(annotation_path) as f:
        annotations = json.load(f)

    labels = np.zeros(shape, dtype="uint8")
    for shape_annotation in annotations["shapes"]:
        fdi_id = int(shape_annotation["label"])
        points = np.asarray(shape_annotation["points"])
        c, r = points[:, 0], points[:, 1]
        rr, cc = polygon(r, c, shape=shape)
        labels[rr, cc] = fdi_id

    return labels


def get_sts2024_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the STS2024 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_sts2024_data(path, download)

    image_dir = os.path.join(data_dir, "Images")
    annotation_dir = os.path.join(data_dir, "Masks")
    preprocessed_dir = os.path.join(data_dir, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    image_paths = natsorted(glob(os.path.join(image_dir, "*.jpg")))

    gt_paths = []
    for image_path in tqdm(image_paths, desc="Preprocessing labels"):
        fname = os.path.splitext(os.path.basename(image_path))[0]
        annotation_path = os.path.join(annotation_dir, f"{fname}_Mask.json")
        gt_path = os.path.join(preprocessed_dir, f"{fname}.tif")

        if not os.path.exists(gt_path):
            image_shape = imageio.imread(image_path).shape[:2]
            labels = _rasterize_annotations(image_shape, annotation_path)
            imageio.imwrite(gt_path, labels)

        gt_paths.append(gt_path)

    return image_paths, gt_paths


def get_sts2024_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the STS2024 dataset for FDI tooth segmentation in panoramic dental radiographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_sts2024_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_sts2024_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the STS2024 dataloader for FDI tooth segmentation in panoramic dental radiographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_sts2024_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
