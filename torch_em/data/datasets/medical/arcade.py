"""The ARCADE dataset contains annotations for coronary artery segmentation and stenosis
detection in X-ray coronary angiography images.

The dataset provides two tasks, each with 1000 training, 200 validation and 300 test images:
- 'syntax': multiclass segmentation of 25 coronary artery segments, following the SYNTAX score
  methodology (see `SYNTAX_LABELS`).
- 'stenosis': binary segmentation of atherosclerotic plaques (stenotic lesions).

The dataset is located at https://zenodo.org/records/10390295 (DOI: 10.5281/zenodo.10390295)
and is distributed under the CC0 1.0 license.

This dataset is from the publication https://doi.org/10.1038/s41597-023-02871-z.
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


URL = "https://zenodo.org/records/10390295/files/arcade.zip"
CHECKSUM = "a396cdea7c92c55dc97bbf3dd8e3df517d76872b289a8bcb45513bdb3350837f"

SYNTAX_LABELS = {
    0: "background", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "9a",
    11: "10", 12: "10a", 13: "11", 14: "12", 15: "12a", 16: "13", 17: "14", 18: "14a", 19: "15", 20: "16",
    21: "16a", 22: "16b", 23: "16c", 24: "12b", 25: "14b",
}
"""The label ids of the coronary artery segment classes for the 'syntax' task, following the SYNTAX
score segment definitions (https://syntaxscore.org/index.php/tutorial/definitions/14-appendix-i-segment-definitions).
"""

STENOSIS_LABELS = {0: "background", 1: "stenosis"}
"""The label ids for the 'stenosis' task. The raw annotations use a single category id (26) for
all stenotic lesions, which is remapped to label id 1 here."""


def get_arcade_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ARCADE dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "arcade")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "arcade.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _rasterize_annotations(shape, annotations, task):
    labels = np.zeros(shape, dtype="uint8")
    for ann in annotations:
        # The 'stenosis' task only has one category (id 26, 'stenosis'), which is remapped to label 1.
        # The 'syntax' task categories (ids 1-25) already match the desired label ids.
        class_id = 1 if task == "stenosis" else ann["category_id"]
        for seg in ann["segmentation"]:
            c, r = np.asarray(seg[0::2]), np.asarray(seg[1::2])
            rr, cc = polygon(r, c, shape=shape)
            labels[rr, cc] = class_id
    return labels


def _preprocess_split(image_dir, annotation_path, preprocessed_dir, task):
    os.makedirs(preprocessed_dir, exist_ok=True)

    with open(annotation_path) as f:
        annotations = json.load(f)

    annotations_by_image = {}
    for ann in annotations["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    image_paths, gt_paths = [], []
    for image_info in tqdm(annotations["images"], desc=f"Preprocessing labels for {image_dir}"):
        image_path = os.path.join(image_dir, image_info["file_name"])
        if not os.path.exists(image_path):
            continue

        gt_path = os.path.join(preprocessed_dir, image_info["file_name"])
        if not os.path.exists(gt_path):
            image_annotations = annotations_by_image.get(image_info["id"], [])
            shape = (image_info["height"], image_info["width"])
            labels = _rasterize_annotations(shape, image_annotations, task)
            imageio.imwrite(gt_path, labels)

        image_paths.append(image_path)
        gt_paths.append(gt_path)

    return image_paths, gt_paths


def get_arcade_paths(
    path: Union[os.PathLike, str],
    task: Literal["syntax", "stenosis"],
    split: Literal["train", "val", "test"],
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the ARCADE data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        task: The choice of task, either 'syntax' (multiclass vessel segment segmentation) or
            'stenosis' (binary stenosis segmentation).
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if task not in ("syntax", "stenosis"):
        raise ValueError(f"'{task}' is not a valid task. Please choose either 'syntax' or 'stenosis'.")

    if split not in ("train", "val", "test"):
        raise ValueError(f"'{split}' is not a valid split.")

    data_dir = get_arcade_data(path, download)

    image_dir = os.path.join(data_dir, task, split, "images")
    annotation_path = os.path.join(data_dir, task, split, "annotations", f"{split}.json")
    preprocessed_dir = os.path.join(data_dir, "preprocessed", task, split)

    if os.path.exists(preprocessed_dir) and len(glob(os.path.join(preprocessed_dir, "*.png"))) > 0:
        image_paths = natsorted(glob(os.path.join(image_dir, "*.png")))
        gt_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.png")))
        return image_paths, gt_paths

    image_paths, gt_paths = _preprocess_split(image_dir, annotation_path, preprocessed_dir, task)
    return natsorted(image_paths), natsorted(gt_paths)


def get_arcade_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    task: Literal["syntax", "stenosis"],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ARCADE dataset for coronary artery and stenosis segmentation in X-ray angiography.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        task: The choice of task, either 'syntax' (multiclass vessel segment segmentation) or
            'stenosis' (binary stenosis segmentation).
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_arcade_paths(path, task, split, download)

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


def get_arcade_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    task: Literal["syntax", "stenosis"],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ARCADE dataloader for coronary artery and stenosis segmentation in X-ray angiography.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        task: The choice of task, either 'syntax' (multiclass vessel segment segmentation) or
            'stenosis' (binary stenosis segmentation).
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_arcade_dataset(path, patch_shape, task, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
