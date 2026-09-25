"""The FracAtlas dataset contains annotations for fracture segmentation in musculoskeletal radiographs.

The dataset consists of 4,083 radiographs of the hand, leg, hip and shoulder, of which 719 fractured
images come with fracture segmentation polygons (COCO annotations with the single category 'fractured', up to
several polygons per image). The remaining 3,366 images are non-fractured and have no annotations, so this
loader only exposes the 719 annotated images. The polygons are rasterized into binary masks
(1 = fracture) during preprocessing, and the images are stored as single-channel tif files, since the JPEGs
are grayscale but a part of them is saved with three identical channels. The image sizes vary from
454x373 to 2880x2304 pixels, use `resize_inputs=True` to train with batches.

The official splits ('train', 'val' and 'test', from 'Utilities/Fracture Split') are available via the
`split` argument. They cover exactly the annotated images.

The dataset is located at https://doi.org/10.6084/m9.figshare.22363012, released under a CC-BY-4.0 license.

This dataset is from the publication https://doi.org/10.1038/s41597-023-02432-4.
Please cite it if you use this dataset for your research.
"""

import os
import json
import uuid
from tqdm import tqdm
from typing import Union, Tuple, Literal, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/65518038"
CHECKSUM = "b67ec2d290a022b3dcf47f78e9a37f7edcc80592c0571f439355bf00bd9f0e23"

SPLITS = ["train", "val", "test"]
SPLIT_FILES = {"train": "train.csv", "val": "valid.csv", "test": "test.csv"}


def _write_atomic(path, array):
    tmp_path = f"{path}.{uuid.uuid4().hex}.incomplete.tif"
    imageio.imwrite(tmp_path, array, extension=".tif")
    os.replace(tmp_path, path)


def _preprocess_data(data_dir, preprocessed_dir):
    from skimage.draw import polygon

    with open(os.path.join(data_dir, "Annotations", "COCO JSON", "COCO_fracture_masks.json")) as f:
        coco = json.load(f)

    image_dir = os.path.join(preprocessed_dir, "images")
    label_dir = os.path.join(preprocessed_dir, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    polygons = {}
    for annotation in coco["annotations"]:
        polygons.setdefault(annotation["image_id"], []).extend(annotation["segmentation"])

    for image_info in tqdm(coco["images"], desc="Preprocess FracAtlas"):
        if image_info["id"] not in polygons:
            continue

        stem = os.path.splitext(image_info["file_name"])[0]
        image_path = os.path.join(image_dir, f"{stem}.tif")
        label_path = os.path.join(label_dir, f"{stem}.tif")
        if os.path.exists(image_path) and os.path.exists(label_path):
            continue

        image = imageio.imread(os.path.join(data_dir, "images", "Fractured", image_info["file_name"]))
        if image.ndim == 3:
            image = image[..., 0]

        label = np.zeros(image.shape, dtype="uint8")
        for coordinates in polygons[image_info["id"]]:
            coordinates = np.asarray(coordinates, dtype="float64").reshape(-1, 2)
            rr, cc = polygon(coordinates[:, 1], coordinates[:, 0], shape=label.shape)
            label[rr, cc] = 1

        _write_atomic(label_path, label)
        _write_atomic(image_path, image)


def get_fracatlas_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the FracAtlas dataset and rasterize the fracture polygons into masks.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the extracted dataset.
    """
    data_dir = os.path.join(path, "FracAtlas")
    if not os.path.exists(data_dir):
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, "FracAtlas.zip")
        util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip(zip_path=zip_path, dst=path, remove=False)

    _preprocess_data(data_dir, os.path.join(path, "preprocessed"))
    return data_dir


def get_fracatlas_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the FracAtlas data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose one of {SPLITS}.")

    data_dir = get_fracatlas_data(path, download)

    with open(os.path.join(data_dir, "Utilities", "Fracture Split", SPLIT_FILES[split])) as f:
        names = [os.path.splitext(line.strip())[0] for line in f.read().splitlines()[1:] if line.strip()]

    preprocessed_dir = os.path.join(path, "preprocessed")
    raw_paths = [os.path.join(preprocessed_dir, "images", f"{name}.tif") for name in sorted(names)]
    label_paths = [os.path.join(preprocessed_dir, "labels", f"{name}.tif") for name in sorted(names)]

    assert len(raw_paths) > 0
    assert all(os.path.exists(p) for p in raw_paths + label_paths)

    return raw_paths, label_paths


def get_fracatlas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the FracAtlas dataset for fracture segmentation in musculoskeletal radiographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_fracatlas_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_fracatlas_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the FracAtlas dataloader for fracture segmentation in musculoskeletal radiographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_fracatlas_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
