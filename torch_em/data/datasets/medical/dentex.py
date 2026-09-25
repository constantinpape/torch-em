"""The DENTEX dataset contains annotations for semantic segmentation of abnormal teeth by their
diagnosis (caries, deep caries, periapical lesion or impacted tooth) in panoramic dental X-rays.

The dataset was curated for the DENTEX 2023 MICCAI challenge (https://dentex.grand-challenge.org),
which comprises panoramic dental X-rays from three institutions, annotated for the quadrant,
tooth (FDI enumeration) and diagnosis of each abnormal tooth. The original annotations are per-tooth
polygons with a quadrant, tooth enumeration and diagnosis label each; this module rasterizes them
into a per-pixel label map of the diagnosis class (see `DIAGNOSIS_CLASSES`), which is the part of
the dataset that adds value over the other, purely structural, tooth segmentation datasets in this
package. It only uses the fully annotated 'quadrant-enumeration-diagnosis' subset: the training
split has ground truth for all classes, and the validation split ground truth was released after
the challenge. The test split annotations were never released and so are not covered here.

The data is hosted on Hugging Face at https://huggingface.co/datasets/ibrahimhamamci/DENTEX
and is distributed under the CC BY-NC-SA 4.0 license.

The dataset is from the publication https://doi.org/10.48550/arXiv.2305.19112.
Please cite it if you use this dataset for your research.
"""

import os
import json
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import numpy as np
from skimage.draw import polygon

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "train": "https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/DENTEX/training_data.zip",
    "val_images": "https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/DENTEX/validation_data.zip",
    "val_labels": "https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/DENTEX/validation_triple.json",
}

CHECKSUMS = {
    "train": "18b2a2dbc5a2b10b0cc6a7677c46a382f4709ab8c9c3bb94f57b74e38e11ffd3",
    "val_images": "6370bb4f1024bd610cde13242a465cb2eff195fc02f56ac22126555e7edc7bc3",
    "val_labels": "d058afd35d2849923c7c045e61fd3e05d231dcf74d55009993fff88bd9b6f5a2",
}

DIAGNOSIS_CLASSES = {0: "background", 1: "impacted", 2: "caries", 3: "periapical_lesion", 4: "deep_caries"}
"""The label ids of the diagnosis classes. The 'category_id_3' in the annotations (0-indexed,
without background) is mapped to the label id (1-indexed, i.e. + 1) so that 0 marks background."""


def _extract_subset(zip_path, path, subdir):
    import zipfile

    with zipfile.ZipFile(zip_path) as f:
        members = [m for m in f.namelist() if f"/{subdir}/" in m and "ipynb_checkpoints" not in m]
        f.extractall(path, members=members)


def get_dentex_data(
    path: Union[os.PathLike, str], split: Literal["train", "val"] = None, download: bool = False
) -> str:
    """Download the DENTEX dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to download. Either 'train' or 'val'. By default downloads both.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    os.makedirs(path, exist_ok=True)

    if split in (None, "train"):
        # NOTE: The training archive names this folder with hyphens ('quadrant-enumeration-disease'),
        # unlike the validation archive, which uses underscores ('quadrant_enumeration_disease').
        train_dir = os.path.join(path, "training_data", "quadrant-enumeration-disease")
        if not os.path.exists(train_dir):
            zip_path = os.path.join(path, "training_data.zip")
            util.download_source(path=zip_path, url=URLS["train"], download=download, checksum=CHECKSUMS["train"])
            _extract_subset(zip_path, path, "quadrant-enumeration-disease")
            os.remove(zip_path)

    if split in (None, "val"):
        val_dir = os.path.join(path, "validation_data", "quadrant_enumeration_disease")
        if not os.path.exists(val_dir):
            zip_path = os.path.join(path, "validation_data.zip")
            util.download_source(
                path=zip_path, url=URLS["val_images"], download=download, checksum=CHECKSUMS["val_images"]
            )
            _extract_subset(zip_path, path, "quadrant_enumeration_disease")
            os.remove(zip_path)

        val_labels_path = os.path.join(path, "validation_triple.json")
        util.download_source(
            path=val_labels_path, url=URLS["val_labels"], download=download, checksum=CHECKSUMS["val_labels"]
        )

    return path


def _rasterize_annotations(shape, annotations):
    labels = np.zeros(shape, dtype="uint8")
    for ann in annotations:
        class_id = ann["category_id_3"] + 1
        for seg in ann["segmentation"]:
            c, r = np.asarray(seg[0::2]), np.asarray(seg[1::2])
            rr, cc = polygon(r, c, shape=shape)
            labels[rr, cc] = class_id
    return labels


def _preprocess_split(image_dir, annotation_path, preprocessed_dir):
    os.makedirs(preprocessed_dir, exist_ok=True)

    with open(annotation_path) as f:
        annotations = json.load(f)

    image_paths, gt_paths = [], []
    for image_info in tqdm(annotations["images"], desc=f"Preprocessing labels for {image_dir}"):
        image_path = os.path.join(image_dir, image_info["file_name"])
        if not os.path.exists(image_path):
            continue

        gt_path = os.path.join(preprocessed_dir, f"{os.path.splitext(image_info['file_name'])[0]}.tif")
        if not os.path.exists(gt_path):
            image_annotations = [a for a in annotations["annotations"] if a["image_id"] == image_info["id"]]
            shape = (image_info["height"], image_info["width"])
            labels = _rasterize_annotations(shape, image_annotations)
            imageio.imwrite(gt_path, labels)

        image_paths.append(image_path)
        gt_paths.append(gt_path)

    return image_paths, gt_paths


def get_dentex_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the DENTEX data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in ("train", "val"):
        raise ValueError(f"'{split}' is not a valid split. Please choose either 'train' or 'val'.")

    data_dir = get_dentex_data(path, split, download)

    if split == "train":
        image_dir = os.path.join(data_dir, "training_data", "quadrant-enumeration-disease", "xrays")
        annotation_path = os.path.join(
            data_dir, "training_data", "quadrant-enumeration-disease", "train_quadrant_enumeration_disease.json"
        )
    else:
        image_dir = os.path.join(data_dir, "validation_data", "quadrant_enumeration_disease", "xrays")
        annotation_path = os.path.join(data_dir, "validation_triple.json")

    preprocessed_dir = os.path.join(data_dir, "preprocessed", split)
    image_paths, gt_paths = _preprocess_split(image_dir, annotation_path, preprocessed_dir)

    image_paths, gt_paths = natsorted(image_paths), natsorted(gt_paths)
    return image_paths, gt_paths


def get_dentex_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DENTEX dataset for tooth diagnosis segmentation in panoramic dental radiographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'val'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_dentex_paths(path, split, download)

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


def get_dentex_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DENTEX dataloader for tooth diagnosis segmentation in panoramic dental radiographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'val'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_dentex_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
