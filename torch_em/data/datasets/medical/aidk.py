"""AIDK is a dataset for the segmentation of the cornea, keratitis lesions and the iris in
anterior-segment optical coherence tomography (AS-OCT) images.

The dataset contains 1,168 AS-OCT images from 64 keratitis patients: 400 'partial-frame' images
(annotated for cornea and lesion) and 768 'full-frame' images (annotated for cornea, lesion and iris).

The dataset is located at https://doi.org/10.6084/m9.figshare.c.7036994.v1 and is distributed under
the CC0 license. The dataset is from the publication https://doi.org/10.1038/s41597-024-03464-0.
Please cite it if you use this dataset for your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np
from skimage import draw
import imageio.v3 as imageio

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/46760137"
CHECKSUM = "70a0227324c288662fb8b02cc67a40facfc4615b085bf9631a965c1f8ed9c921"

FRAMES = ["partial", "full"]
TASKS = ["cornea", "lesion", "iris"]
LABELS = {"cornea": "Cornea", "lesion": "Lesion", "iris": "Iris"}


def get_aidk_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the AIDK dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "AIDK_Dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "AIDK_Dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _create_mask(annotation_path, image_shape, label):
    with open(annotation_path) as f:
        annotation = json.load(f)

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for shape in annotation["shapes"]:
        if shape["label"] != label or shape["shape_type"] != "polygon":
            continue

        points = np.array(shape["points"])
        rr, cc = draw.polygon(points[:, 1], points[:, 0], shape=mask.shape)
        mask[rr, cc] = 1

    return mask


def _preprocess_labels(data_dir, task, frame):
    label = LABELS[task]
    # Only the full-frame images carry iris annotations.
    frame_names = ["full"] if task == "iris" else ([frame] if frame != "all" else FRAMES)

    image_paths, gt_paths = [], []
    for frame_name in frame_names:
        frame_dir = os.path.join(data_dir, f"{frame_name.capitalize()}-frame_Dataset")
        image_dir = os.path.join(frame_dir, "Original_AS-OCT_Images")
        annotation_dir = os.path.join(frame_dir, "Experts_Annotations")
        gt_dir = os.path.join(frame_dir, f"masks_{task}")
        os.makedirs(gt_dir, exist_ok=True)

        annotation_paths = natsorted(
            p for p in glob(os.path.join(annotation_dir, "*.json")) if not os.path.basename(p).startswith("._")
        )

        for annotation_path in tqdm(annotation_paths, desc=f"Converting '{frame_name}-frame' annotations to masks"):
            image_id = Path(annotation_path).stem
            image_path = os.path.join(image_dir, f"{image_id}.bmp")
            if not os.path.exists(image_path):
                continue

            gt_path = os.path.join(gt_dir, f"{image_id}.tif")
            if not os.path.exists(gt_path):
                image_shape = imageio.imread(image_path).shape
                mask = _create_mask(annotation_path, image_shape, label)
                imageio.imwrite(gt_path, mask)

            image_paths.append(image_path)
            gt_paths.append(gt_path)

    return image_paths, gt_paths


def get_aidk_paths(
    path: Union[os.PathLike, str],
    task: Literal["cornea", "lesion", "iris"] = "cornea",
    frame: Literal["partial", "full", "all"] = "all",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the AIDK data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        task: The choice of segmentation task. Either 'cornea', 'lesion' or 'iris'.
        frame: The choice of image subset. Either 'partial', 'full' or 'all'. Ignored (forced to 'full')
            when `task` is 'iris', as only the full-frame images have iris annotations.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if task not in TASKS:
        raise ValueError(f"'{task}' is not a valid task. Please choose one of {TASKS}.")
    if frame not in FRAMES + ["all"]:
        raise ValueError(f"'{frame}' is not a valid frame choice. Please choose one of {FRAMES + ['all']}.")

    data_dir = get_aidk_data(path=path, download=download)
    image_paths, gt_paths = _preprocess_labels(data_dir, task, frame)

    return image_paths, gt_paths


def get_aidk_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    task: Literal["cornea", "lesion", "iris"] = "cornea",
    frame: Literal["partial", "full", "all"] = "all",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Get the AIDK dataset for segmentation of the cornea, keratitis lesions and the iris in AS-OCT images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        task: The choice of segmentation task. Either 'cornea', 'lesion' or 'iris'.
        frame: The choice of image subset. Either 'partial', 'full' or 'all'. Ignored (forced to 'full')
            when `task` is 'iris'.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_aidk_paths(path, task, frame, download)

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


def get_aidk_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    task: Literal["cornea", "lesion", "iris"] = "cornea",
    frame: Literal["partial", "full", "all"] = "all",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Get the AIDK dataloader for segmentation of the cornea, keratitis lesions and the iris in AS-OCT images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        task: The choice of segmentation task. Either 'cornea', 'lesion' or 'iris'.
        frame: The choice of image subset. Either 'partial', 'full' or 'all'. Ignored (forced to 'full')
            when `task` is 'iris'.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_aidk_dataset(path, patch_shape, task, frame, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
