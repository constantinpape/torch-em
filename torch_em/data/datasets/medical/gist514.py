"""The GIST514-DB dataset contains annotations for lesion segmentation in endoscopic ultrasound
(EUS) images of gastrointestinal stromal tumors (GISTs) and leiomyomas.

The dataset comprises 514 EUS cases (251 GIST, 263 leiomyoma), with lesion contours manually
delineated and verified by expert clinicians. It is the dataset introduced for the 'Query2'
GIST detection framework.

The data is hosted on Google Drive, linked from https://github.com/howardchina/query2, and is
distributed under the CC BY-NC-SA 4.0 license.

The dataset is from the publication https://doi.org/10.1016/j.compbiomed.2022.106424.
Please cite it if you use this dataset for your research.
"""

import os
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import json
import numpy as np
from skimage.draw import polygon

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://drive.google.com/drive/folders/1TG9Bq-OaKkMXV2s42f_oZJdoTOfIAZLi"

CATEGORIES = {0: "background", 1: "leiomyoma", 2: "GIST"}
"""The label ids of the lesion categories. The 'category_id' in the annotations (1: leiomyoma,
2: GIST) is used directly as the pixel label, so that 0 marks background."""


def get_gist514_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the GIST514-DB dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "usd514-db", "usd514_jpeg_roi")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    util.download_source_gdrive(path=path, url=URL, download=download, download_type="folder", expected_samples=1200)

    return data_dir


def _rasterize_annotations(shape, image_annotations):
    labels = np.zeros(shape, dtype="uint8")
    for ann in image_annotations:
        category_id = ann["category_id"]
        for seg in ann["segmentation"]:
            c, r = np.asarray(seg[0::2]), np.asarray(seg[1::2])
            rr, cc = polygon(r, c, shape=shape)
            labels[rr, cc] = category_id
    return labels


def get_gist514_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the GIST514-DB data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_gist514_data(path, download)

    image_dir = os.path.join(data_dir, "images")
    annotation_path = os.path.join(data_dir, "annotations", "all_anno_crop.json")
    preprocessed_dir = os.path.join(data_dir, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    with open(annotation_path) as f:
        annotations = json.load(f)

    image_paths, gt_paths = [], []
    for image_info in tqdm(annotations["images"], desc="Preprocessing labels"):
        image_path = os.path.join(image_dir, image_info["file_name"])
        if not os.path.exists(image_path):
            continue

        fname = os.path.splitext(image_info["file_name"])[0]
        gt_path = os.path.join(preprocessed_dir, f"{fname}.tif")
        if not os.path.exists(gt_path):
            image_annotations = [a for a in annotations["annotations"] if a["image_id"] == image_info["id"]]
            shape = (image_info["height"], image_info["width"])
            labels = _rasterize_annotations(shape, image_annotations)
            imageio.imwrite(gt_path, labels)

        image_paths.append(image_path)
        gt_paths.append(gt_path)

    image_paths, gt_paths = natsorted(image_paths), natsorted(gt_paths)
    return image_paths, gt_paths


def get_gist514_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the GIST514-DB dataset for lesion segmentation in endoscopic ultrasound images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_gist514_paths(path, download)

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


def get_gist514_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the GIST514-DB dataloader for lesion segmentation in endoscopic ultrasound images.

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
    dataset = get_gist514_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
