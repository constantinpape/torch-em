"""The Plantorganelle Hunter dataset contains annotations for semantic segmentation of chloroplasts, mitochondria,
vacuoles and nuclei in transmission electron microscopy (TEM) images of plant cells.

There are 902 large 2D micrographs (mostly 4800x3230 pixels), split into 541 train, 180 val and 181 test images.

The dataset is from the publication "Plantorganelle Hunter is an effective deep-learning-based method for plant
organelle phenotyping in electron microscopy" and is available at https://doi.org/10.57760/sciencedb.01335.
Please cite it if you use this dataset in your research.
"""

import os
from functools import partial
from typing import List, Literal, Optional, Tuple, Union

import imageio.v3 as imageio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


# NOTE: The data is hosted on ScienceDB, which does not provide stable direct download links.
# The full archive is ~50 GB, but most of it is not used here (3D stacks, unseen data, ...).
DOI = "https://doi.org/10.57760/sciencedb.01335"

# The mask files are palette images, with one (VOC-style) color per class.
CLASSES = {"chloroplast": 1, "mitochondria": 2, "vacuole": 3, "nucleus": 4}
COLORS_TO_CLASSES = {(128, 0, 0): 1, (0, 128, 0): 2, (128, 128, 0): 3, (0, 0, 128): 4}


def _convert_label(label_path, out_path):
    # We have to convert the labels, because the palette images get loaded as multi-channel RGB images.
    mask = Image.open(label_path)
    palette = np.array(mask.getpalette(), dtype="uint8").reshape(-1, 3)
    mask = np.array(mask)

    lut = np.zeros(256, dtype="uint8")
    for idx in np.unique(mask):
        color = tuple(int(c) for c in palette[idx])
        if color == (0, 0, 0):
            continue
        if color not in COLORS_TO_CLASSES:
            raise ValueError(f"Unexpected label color {color} in {label_path}.")
        lut[idx] = COLORS_TO_CLASSES[color]

    imageio.imwrite(out_path, lut[mask], compression=9)


def get_plantorganelle_hunter_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Prepare the Plantorganelle Hunter dataset.

    The data cannot be downloaded automatically. Please download the archive from
    https://doi.org/10.57760/sciencedb.01335 and extract the folders 'image', 'label' and 'splits' to `path`.

    Args:
        path: Filepath to the folder with the extracted data.
        download: Whether to download the data if it is not present. This is not supported for this dataset.

    Returns:
        The filepath to the folder with the converted label masks.
    """
    label_dir = os.path.join(path, "label_classes")
    if os.path.exists(os.path.join(label_dir, ".done")):
        return label_dir

    missing = [name for name in ("image", "label", "splits") if not os.path.exists(os.path.join(path, name))]
    if missing:
        raise RuntimeError(
            "The Plantorganelle Hunter dataset cannot be downloaded automatically. "
            f"Please download it from {DOI} and extract the folders 'image', 'label' and 'splits' to '{path}'. "
            f"Currently missing: {missing}."
        )

    os.makedirs(label_dir, exist_ok=True)
    for split in ("train", "val", "test"):
        for name in _read_split(path, split):
            out_path = os.path.join(label_dir, f"{name}.png")
            if not os.path.exists(out_path):
                _convert_label(os.path.join(path, "label", f"{name}.png"), out_path)

    open(os.path.join(label_dir, ".done"), "w").close()
    return label_dir


def _read_split(path, split):
    with open(os.path.join(path, "splits", f"{split}.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def get_plantorganelle_hunter_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Plantorganelle Hunter data.

    Args:
        path: Filepath to the folder with the extracted data.
        split: The data split. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present. This is not supported for this dataset.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"'{split}' is not a valid split. Please choose from 'train' / 'val' / 'test'.")

    label_dir = get_plantorganelle_hunter_data(path, download)

    names = _read_split(path, split)
    raw_paths = [os.path.join(path, "image", f"{name}.tif") for name in names]
    label_paths = [os.path.join(label_dir, f"{name}.png") for name in names]
    assert raw_paths and all(os.path.exists(p) for p in raw_paths), "Some of the images in the split are missing."

    return raw_paths, label_paths


def _select_class(labels, class_id):
    return (labels == class_id).astype("uint8")


def get_plantorganelle_hunter_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    label_choice: Optional[str] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Plantorganelle Hunter dataset for organelle segmentation in plant cells.

    Args:
        path: Filepath to the folder with the extracted data.
        patch_shape: The patch shape to use for training.
        split: The data split. Either 'train', 'val' or 'test'.
        label_choice: The structure to segment. One of 'chloroplast', 'mitochondria', 'vacuole' or 'nucleus'.
            By default (None), all structures are returned as a semantic segmentation
            (0: background, 1: chloroplast, 2: mitochondria, 3: vacuole, 4: nucleus).
        download: Whether to download the data if it is not present. This is not supported for this dataset.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if label_choice is not None:
        if label_choice not in CLASSES:
            raise ValueError(f"'{label_choice}' is not a valid label choice. Please choose from {list(CLASSES)}.")
        kwargs = util.update_kwargs(kwargs, "label_transform", partial(_select_class, class_id=CLASSES[label_choice]))

    raw_paths, label_paths = get_plantorganelle_hunter_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_plantorganelle_hunter_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    label_choice: Optional[str] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Plantorganelle Hunter dataloader for organelle segmentation in plant cells.

    Args:
        path: Filepath to the folder with the extracted data.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split. Either 'train', 'val' or 'test'.
        label_choice: The structure to segment. One of 'chloroplast', 'mitochondria', 'vacuole' or 'nucleus'.
            By default (None), all structures are returned as a semantic segmentation
            (0: background, 1: chloroplast, 2: mitochondria, 3: vacuole, 4: nucleus).
        download: Whether to download the data if it is not present. This is not supported for this dataset.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_plantorganelle_hunter_dataset(path, patch_shape, split, label_choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
