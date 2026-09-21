"""The ChestX-Det dataset contains annotations for segmentation of 13 categories of thoracic
abnormalities or diseases in chest x-ray images.

The dataset consists of 3578 images from NIH ChestX-14, annotated by three board-certified
radiologists with polygon contours for the 13 categories (see `CHESTX_DET_LABELS`). The dataset
is located at https://github.com/Deepwise-AILab/ChestX-Det-Dataset and is distributed under the
Apache 2.0 license.

This dataset is from the publication https://doi.org/10.48550/arXiv.2004.10871.
Please cite it if you use this dataset for your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import numpy as np
from skimage.draw import polygon
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = {
    "images": {
        "train": "http://resource.deepwise.com/ChestX-Det/train_data.zip",
        "test": "http://resource.deepwise.com/ChestX-Det/test_data.zip",
    },
    "annotations": {
        "train": "https://raw.githubusercontent.com/Deepwise-AILab/ChestX-Det-Dataset/main/ChestX_Det_train.json",
        "test": "https://raw.githubusercontent.com/Deepwise-AILab/ChestX-Det-Dataset/main/ChestX_Det_test.json",
    },
}

CHECKSUM = {
    "images": {
        "train": "413f74a03383280e2d63f6215c8eb581aa386cfec323fb4148f0916d2f5f2900",
        "test": "c52677d1e4043bf425bc997d62db2a780f6260ea1d67ad37c65fe3b2ffdef14f",
    },
    "annotations": {
        "train": None,
        "test": None,
    },
}

CHESTX_DET_LABELS = {
    0: "background",
    1: "Atelectasis",
    2: "Calcification",
    3: "Cardiomegaly",
    4: "Consolidation",
    5: "Diffuse Nodule",
    6: "Effusion",
    7: "Emphysema",
    8: "Fibrosis",
    9: "Fracture",
    10: "Mass",
    11: "Nodule",
    12: "Pleural Thickening",
    13: "Pneumothorax",
}
"""The label ids of the 13 categories of thoracic abnormalities or diseases annotated in ChestX-Det."""

LABEL_IDS = {name: label_id for label_id, name in CHESTX_DET_LABELS.items() if label_id != 0}


def get_chestx_det_data(path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False) -> str:
    """Download the ChestX-Det data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the image data is downloaded.
    """
    if split not in ("train", "test"):
        raise ValueError(f"'{split}' is not a valid split.")

    image_dir = os.path.join(path, split)
    if os.path.exists(image_dir):
        return image_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"{split}_data.zip")
    util.download_source(path=zip_path, url=URL["images"][split], download=download, checksum=CHECKSUM["images"][split])
    util.unzip(zip_path=zip_path, dst=path, remove=False)

    annotation_path = os.path.join(path, f"ChestX_Det_{split}.json")
    util.download_source(
        path=annotation_path, url=URL["annotations"][split], download=download, checksum=CHECKSUM["annotations"][split]
    )

    return image_dir


def _rasterize_annotations(shape, syms, polygons):
    labels = np.zeros(shape, dtype="uint8")
    for sym, poly in zip(syms, polygons):
        poly = np.asarray(poly)
        rr, cc = polygon(poly[:, 1], poly[:, 0], shape=shape)
        labels[rr, cc] = LABEL_IDS[sym]
    return labels


def _preprocess_split(image_dir, annotation_path, preprocessed_dir):
    os.makedirs(preprocessed_dir, exist_ok=True)

    with open(annotation_path) as f:
        annotations = json.load(f)

    image_paths, gt_paths = [], []
    for ann in tqdm(annotations, desc=f"Preprocessing labels for {image_dir}"):
        image_path = os.path.join(image_dir, ann["file_name"])
        if not os.path.exists(image_path):
            continue

        gt_path = os.path.join(preprocessed_dir, ann["file_name"])
        if not os.path.exists(gt_path):
            shape = imageio.imread(image_path).shape[:2]
            labels = _rasterize_annotations(shape, ann["syms"], ann["polygons"])
            imageio.imwrite(gt_path, labels)

        image_paths.append(image_path)
        gt_paths.append(gt_path)

    return image_paths, gt_paths


def get_chestx_det_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the ChestX-Det data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    image_dir = get_chestx_det_data(path=path, split=split, download=download)

    annotation_path = os.path.join(path, f"ChestX_Det_{split}.json")
    preprocessed_dir = os.path.join(path, "preprocessed", split)

    if os.path.exists(preprocessed_dir) and len(glob(os.path.join(preprocessed_dir, "*.png"))) > 0:
        image_paths = natsorted(glob(os.path.join(image_dir, "*.png")))
        gt_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.png")))
        return image_paths, gt_paths

    image_paths, gt_paths = _preprocess_split(image_dir, annotation_path, preprocessed_dir)
    return natsorted(image_paths), natsorted(gt_paths)


def get_chestx_det_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ChestX-Det dataset for segmentation of thoracic abnormalities in chest x-rays.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_chestx_det_paths(path, split, download)

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
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_chestx_det_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ChestX-Det dataloader for segmentation of thoracic abnormalities in chest x-rays.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_chestx_det_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
