"""The Fluorescent Neuronal Cells v2 dataset contains annotations for cell segmentation
in fluorescence microscopy images of rodent brain slices.

The dataset holds three image collections, named after the marker color: green, yellow and red.
Each collection has a trainval and a test split. The archive ships a binary mask per image, and a
COCO file that stores one polygon per cell. This loader rasterizes the polygons, so the target is an
instance segmentation.

NOTE: The collections differ a lot. The green images show small punctate nuclei, the yellow images
show sparse bright cells, and the red images show dense filamentous neurons at a larger size.

NOTE: The COCO files of this dataset do not follow the usual layout. They store one annotation per
image, and its 'segmentation', 'bbox', 'dots' and 'area' fields are parallel lists over the cells.

NOTE: The green trainval split ships a mask for '128.png' but not the image, so this loader skips
that sample. It yields 749 labelled images instead of the 750 that the publication reports.

The dataset is located at https://amsacta.unibo.it/id/eprint/7347 under the CC BY 4.0 license.
This dataset is from the publication https://doi.org/10.1038/s41597-024-03005-9.
Please cite it if you use this dataset in your research.
"""

import os
import json
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URLS = {
    "green": "https://amsacta.unibo.it/id/eprint/7347/28/green.zip",
    "yellow": "https://amsacta.unibo.it/id/eprint/7347/27/yellow.zip",
    "red": "https://amsacta.unibo.it/id/eprint/7347/29/red.zip",
}

CHECKSUMS = {
    "green": "7760c3c55d236c17f2225a0a63e8e61a86c93d0645c7188bbe9652f22f520517",
    "yellow": "08048f5c71d4afa726cc54fd887a06cc6894aff62f6704ff5a8c15aa2f9946ca",
    "red": "0a9a2ac9dbdd7222cadf44a7846f61f9d0adb08cf09b801efb917a76e8704533",
}

COLLECTIONS = tuple(URLS)
SPLITS = ("trainval", "test")


def _rasterize(annotation, shape: Tuple[int, int]) -> np.ndarray:
    """Draw one label per cell polygon of an image."""
    from skimage.draw import polygon as draw_polygon

    labels = np.zeros(shape, dtype="uint16")
    for instance_id, part in enumerate(annotation["segmentation"], start=1):
        polygon = np.array(part, dtype=float).reshape(-1, 2)
        rows, columns = draw_polygon(polygon[:, 1], polygon[:, 0], shape=shape)
        labels[rows, columns] = instance_id
    return labels


def _create_instance_labels(data_dir: str, collection: str, split: str) -> str:
    """Rasterize the COCO polygons of one collection split into instance label images."""
    from tqdm import tqdm

    label_dir = os.path.join(data_dir, collection, split, "instance_labels")
    os.makedirs(label_dir, exist_ok=True)

    coco_paths = glob(os.path.join(data_dir, collection, split, "ground_truths", "COCO", "*.json"))
    if not coco_paths:
        raise RuntimeError(f"Could not find the COCO file for '{collection}/{split}' in {data_dir}.")

    with open(coco_paths[0]) as f:
        coco = json.load(f)

    images = {image["id"]: image["file_name"] for image in coco["images"]}
    image_dir = os.path.join(data_dir, collection, split, "images")

    for annotation in tqdm(coco["annotations"], desc=f"Preprocess '{collection}/{split}'"):
        file_name = images.get(annotation["image_id"])
        if file_name is None:
            continue

        # The green trainval split lists an image that the archive does not contain.
        image_path = os.path.join(image_dir, file_name)
        if not os.path.exists(image_path):
            continue

        output_path = os.path.join(label_dir, f"{Path(file_name).stem}.tif")
        if os.path.exists(output_path):
            continue

        shape = imageio.imread(image_path).shape[:2]
        imageio.imwrite(output_path, _rasterize(annotation, shape), compression="zlib")

    return label_dir


def get_fluo_neuronal_cells_data(
    path: Union[os.PathLike, str],
    collection: Literal["green", "yellow", "red"] = "green",
    download: bool = False,
) -> str:
    """Download the Fluorescent Neuronal Cells dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        collection: The image collection. Either 'green', 'yellow' or 'red'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder that holds the collections.
    """
    if collection not in COLLECTIONS:
        raise ValueError(f"'{collection}' is not a valid collection. Choose from {list(COLLECTIONS)}.")

    collection_dir = os.path.join(path, collection)
    if os.path.exists(collection_dir):
        return path

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"{collection}.zip")
    util.download_source(zip_path, URLS[collection], download, CHECKSUMS[collection])
    util.unzip(zip_path=zip_path, dst=path)

    return path


def get_fluo_neuronal_cells_paths(
    path: Union[os.PathLike, str],
    split: Literal["trainval", "test"] = "trainval",
    collection: Optional[Union[str, Sequence[str]]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Fluorescent Neuronal Cells data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'trainval' or 'test'.
        collection: The image collection or collections. Defaults to all of them.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the instance label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLITS)}.")

    if collection is None:
        collections = list(COLLECTIONS)
    else:
        collections = [collection] if isinstance(collection, str) else list(collection)

    image_paths, label_paths = [], []
    for name in collections:
        data_dir = get_fluo_neuronal_cells_data(path, name, download)
        label_dir = _create_instance_labels(data_dir, name, split)

        for label_path in natsorted(glob(os.path.join(label_dir, "*.tif"))):
            image_path = os.path.join(data_dir, name, split, "images", f"{Path(label_path).stem}.png")
            if not os.path.exists(image_path):
                continue
            image_paths.append(image_path)
            label_paths.append(label_path)

    if not image_paths:
        raise RuntimeError(f"Could not find any Fluorescent Neuronal Cells data in {path}.")

    return image_paths, label_paths


def get_fluo_neuronal_cells_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["trainval", "test"] = "trainval",
    collection: Optional[Union[str, Sequence[str]]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Fluorescent Neuronal Cells dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'trainval' or 'test'.
        collection: The image collection or collections. Defaults to all of them.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The Fluorescent Neuronal Cells patch shape must be two-dimensional, got {patch_shape}.")

    image_paths, label_paths = get_fluo_neuronal_cells_paths(path, split, collection, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary,
    )
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        ndim=2,
        **kwargs,
    )


def get_fluo_neuronal_cells_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["trainval", "test"] = "trainval",
    collection: Optional[Union[str, Sequence[str]]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the Fluorescent Neuronal Cells dataloader for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'trainval' or 'test'.
        collection: The image collection or collections. Defaults to all of them.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_fluo_neuronal_cells_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        collection=collection,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
