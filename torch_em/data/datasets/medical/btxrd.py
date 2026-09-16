"""BTXRD is a dataset for the segmentation of primary bone tumors in radiographs.

This dataset is located at https://doi.org/10.6084/m9.figshare.27865398.
The dataset is from the publication https://doi.org/10.1038/s41597-024-04311-y.
Please cite it if you use this dataset for your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
from skimage import draw
import imageio.v3 as imageio

import torch_em

from .. import util
from ..light_microscopy.neurips_cell_seg import to_rgb


URL = "https://ndownloader.figshare.com/files/50653575"
CHECKSUM = "e7c800c3b4e090262b160525a0765f9d93bcc53d639c03806a3ac47b0ded3373"


def get_btxrd_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BTXRD dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "BTXRD")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "BTXRD.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _create_mask(annotation_path, image_shape):
    with open(annotation_path) as f:
        annotation = json.load(f)

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for shape in annotation["shapes"]:
        if shape["shape_type"] != "polygon":
            continue

        points = np.array(shape["points"])
        rr, cc = draw.polygon(points[:, 1], points[:, 0], shape=mask.shape)
        mask[rr, cc] = 1

    return mask


def _preprocess_labels(data_dir):
    image_dir = os.path.join(data_dir, "images")
    annotation_dir = os.path.join(data_dir, "Annotations")
    gt_dir = os.path.join(data_dir, "masks")
    os.makedirs(gt_dir, exist_ok=True)

    annotation_paths = natsorted(glob(os.path.join(annotation_dir, "*.json")))

    image_paths, gt_paths = [], []
    for annotation_path in tqdm(annotation_paths, desc="Converting polygon annotations to masks"):
        image_id = Path(annotation_path).stem

        image_path = glob(os.path.join(image_dir, f"{image_id}.jp*g"))
        assert len(image_path) == 1, f"Could not find a unique matching image for '{image_id}'."
        image_path = image_path[0]

        gt_path = os.path.join(gt_dir, f"{image_id}.tif")
        if not os.path.exists(gt_path):
            image_shape = imageio.imread(image_path).shape
            mask = _create_mask(annotation_path, image_shape)
            imageio.imwrite(gt_path, mask)

        image_paths.append(image_path)
        gt_paths.append(gt_path)

    return image_paths, gt_paths


def get_btxrd_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the BTXRD data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_btxrd_data(path=path, download=download)
    image_paths, gt_paths = _preprocess_labels(data_dir)
    return image_paths, gt_paths


def get_btxrd_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Get the BTXRD dataset for segmentation of primary bone tumors in radiographs.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_btxrd_paths(path=path, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs,
            patch_shape=patch_shape,
            resize_inputs=resize_inputs,
            resize_kwargs=resize_kwargs,
            ensure_rgb=to_rgb,
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


def get_btxrd_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Get the BTXRD dataloader for segmentation of primary bone tumors in radiographs.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_btxrd_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
