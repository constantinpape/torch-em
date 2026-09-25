"""The M-ROSE bacteria dataset contains annotations for bacteria segmentation
in Gram-stained bright-field microscopy images of respiratory specimens.

The images come from microbiological rapid on-site evaluation (M-ROSE) of patients with a lung
infection. The dataset holds 6005 crops of 640 x 640 pixels with 11824 bacteria. Every bacterium
has a polygon and a Gram status, so this loader can return instance labels or the two Gram classes.

NOTE: The annotations are sparse. A clinical smear holds debris and unlabelled objects, and every
object without a polygon becomes background. Take this into account when you measure recall.

NOTE: The archive also holds detection labels that separate cocci from bacilli. This loader uses
the segmentation folder only, and its labels carry the Gram status alone.

NOTE: The image '000077_0_6' is in none of the three split files, so the loader skips it. The splits
cover 6004 of the 6005 crops.

The dataset is located at https://doi.org/10.5281/zenodo.10526360 under the CC BY 4.0 license.
This dataset is from the publication https://doi.org/10.1038/s41597-024-03370-5.
Please cite it if you use this dataset in your research.
"""

import os
import json
import zipfile
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URL = "https://zenodo.org/records/10526360/files/DeepDataSet.zip?download=1"
CHECKSUM = "c5ecaa65fa8c515b3495148c55f48aceeecea0114b4ecbf59c6faf43fb2df4a6"

# The archive also holds a detection folder and a plain image folder, which this loader ignores.
ARCHIVE_FOLDER = "SegmentationDataSet"

SPLITS = ("train", "val", "test")

# The annotations mark a Gram-positive bacterium with 'G+' and a Gram-negative one with 'G'.
GRAM_IDS = {"G+": 1, "G": 2}


def _extract_segmentation_folder(zip_path: str, path: str) -> None:
    """Extract only the segmentation folder of the archive."""
    with zipfile.ZipFile(zip_path) as archive:
        members = [n for n in archive.namelist() if n.startswith(f"{ARCHIVE_FOLDER}/")]
        if not members:
            raise RuntimeError(f"The archive {zip_path} does not hold a '{ARCHIVE_FOLDER}' folder.")
        archive.extractall(path, members=members)


def _rasterize(shapes, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Draw one label per bacterium, and a second image with the Gram status."""
    from skimage.draw import polygon as draw_polygon

    instances = np.zeros(shape, dtype="uint16")
    semantic = np.zeros(shape, dtype="uint8")
    for instance_id, item in enumerate(shapes, start=1):
        points = np.array(item["points"], dtype=float)
        rows, columns = draw_polygon(points[:, 1], points[:, 0], shape=shape)
        instances[rows, columns] = instance_id
        semantic[rows, columns] = GRAM_IDS.get(item.get("label"), 0)
    return instances, semantic


def _create_labels(data_dir: str) -> Tuple[str, str]:
    """Rasterize the polygons of every crop into label images."""
    from tqdm import tqdm

    instance_dir = os.path.join(data_dir, "instance_labels")
    semantic_dir = os.path.join(data_dir, "semantic_labels")
    os.makedirs(instance_dir, exist_ok=True)
    os.makedirs(semantic_dir, exist_ok=True)

    json_paths = natsorted(glob(os.path.join(data_dir, "json", "*.json")))
    for json_path in tqdm(json_paths, desc="Preprocess the M-ROSE annotations"):
        stem = Path(json_path).stem
        instance_path = os.path.join(instance_dir, f"{stem}.tif")
        semantic_path = os.path.join(semantic_dir, f"{stem}.tif")
        if os.path.exists(instance_path) and os.path.exists(semantic_path):
            continue

        with open(json_path) as f:
            annotation = json.load(f)

        shape = (annotation["imageHeight"], annotation["imageWidth"])
        instances, semantic = _rasterize(annotation.get("shapes", []), shape)
        imageio.imwrite(instance_path, instances, compression="zlib")
        imageio.imwrite(semantic_path, semantic, compression="zlib")

    return instance_dir, semantic_dir


def get_mrose_bacteria_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the M-ROSE bacteria dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted segmentation data.
    """
    data_dir = os.path.join(path, ARCHIVE_FOLDER)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "DeepDataSet.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    _extract_segmentation_folder(zip_path, path)

    return data_dir


def get_mrose_bacteria_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"] = "train",
    label_choice: Literal["instances", "semantic"] = "instances",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the M-ROSE bacteria data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train', 'val' or 'test'.
        label_choice: The label to use. Either 'instances' for the bacteria, or 'semantic' for the
            Gram classes, where one is Gram-positive and two is Gram-negative.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLITS)}.")
    if label_choice not in ("instances", "semantic"):
        raise ValueError(f"'{label_choice}' is not a valid label choice. Choose 'instances' or 'semantic'.")

    data_dir = get_mrose_bacteria_data(path, download)
    instance_dir, semantic_dir = _create_labels(data_dir)
    label_dir = instance_dir if label_choice == "instances" else semantic_dir

    split_path = os.path.join(data_dir, "txt", f"{split}.txt")
    with open(split_path) as f:
        stems = [line.strip() for line in f if line.strip()]

    image_paths, label_paths = [], []
    for stem in stems:
        image_path = os.path.join(data_dir, "images", f"{stem}.jpg")
        label_path = os.path.join(label_dir, f"{stem}.tif")
        if not (os.path.exists(image_path) and os.path.exists(label_path)):
            continue
        image_paths.append(image_path)
        label_paths.append(label_path)

    if not image_paths:
        raise RuntimeError(f"Could not find any M-ROSE data for the '{split}' split in {data_dir}.")

    return image_paths, label_paths


def get_mrose_bacteria_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"] = "train",
    label_choice: Literal["instances", "semantic"] = "instances",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the M-ROSE bacteria dataset for bacteria segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'train', 'val' or 'test'.
        label_choice: The label to use. Either 'instances' for the bacteria, or 'semantic' for the
            Gram classes, where one is Gram-positive and two is Gram-negative.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The M-ROSE patch shape must be two-dimensional, got {patch_shape}.")

    image_paths, label_paths = get_mrose_bacteria_paths(path, split, label_choice, download)

    if label_choice == "instances":
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


def get_mrose_bacteria_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"] = "train",
    label_choice: Literal["instances", "semantic"] = "instances",
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the M-ROSE bacteria dataloader for bacteria segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'train', 'val' or 'test'.
        label_choice: The label to use. Either 'instances' for the bacteria, or 'semantic' for the
            Gram classes, where one is Gram-positive and two is Gram-negative.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mrose_bacteria_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        label_choice=label_choice,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
