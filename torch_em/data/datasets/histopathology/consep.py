"""The CoNSeP dataset contains annotations for nucleus segmentation in
H&E stained histopathology images for multi-tissue regions.

NOTE: The source of this dataset is an open-source version hosted on Kaggle:
- https://www.kaggle.com/datasets/rftexas/tiled-consep-224x224px

This dataset is from the publication https://doi.org/10.1016/j.media.2019.101563.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import h5py
import imageio.v3 as imageio
import numpy as np
import torch_em

from elf.segmentation.stitching import stitch_tiled_segmentation
from scipy.io import loadmat
from skimage.measure import label as connected_components
from torch.utils.data import Dataset, DataLoader

from .. import util


def _preprocess_image(raw_paths, label_paths, output_path):

    # Find the start and stop coordinates for all tiles by parsing their filenames.
    tile_coordinates = []
    for path in raw_paths:
        tile_coords = tuple(int(coord) for coord in Path(path).stem.split("_")[2:])
        tile_coordinates.append(tile_coords)

    # Find the dimension of the image as the maximum of the tile coordinates.
    h = max(coord[1] for coord in tile_coordinates)
    w = max(coord[3] for coord in tile_coordinates)
    shape = (h, w)

    # Stitch together the image data.
    raw = np.zeros(shape + (3,), dtype="uint8")
    for path, coords in zip(raw_paths, tile_coordinates):
        tile_data = imageio.imread(path)
        y1, y2, x1, x2 = coords
        raw[y1:y2, x1:x2] = tile_data

    # Stitch together the label data.
    # First, we load the labels and apply an offset so that we have unique ids.
    # Also, some parts of the labels are over-lapping and we make sure to only write it once.
    offset = 0
    labels = np.zeros(shape, dtype="uint32")
    written = np.zeros(shape, dtype=bool)
    for path, coords in zip(label_paths, tile_coordinates):
        y1, y2, x1, x2 = coords

        tile_labels = loadmat(path)["instance_map"]
        tile_labels = connected_components(tile_labels).astype("uint32")

        # Find the mask where we have labels in this tile, and where data was already written.
        tile_mask = tile_labels != 0
        tile_not_written = ~written[y1:y2, x1:x2]

        # And intersect them.
        tile_mask = np.logical_and(tile_mask, tile_not_written)

        # Add up the offset to this tile, unless it is empty.
        if tile_mask.sum() > 0:
            tile_labels[tile_mask] += offset
            offset = int(tile_labels.max())

        # Write out what has been written and the labels.
        written[y1:y2, x1:x2][tile_mask] = 1
        labels[y1:y2, x1:x2][tile_mask] = tile_labels[tile_mask]

    # Stitch the labels together.
    tile_shape = (224, 224)
    stitched_labels = stitch_tiled_segmentation(labels, tile_shape=tile_shape, overlap=1, verbose=False)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("raw", data=raw.transpose(2, 0, 1), compression="gzip")
        f.create_dataset("labels", data=stitched_labels, compression="gzip")


def _preprocess_data(data_dir, split):
    preprocessed_dir = os.path.join(data_dir, "preprocessed", split)
    os.makedirs(preprocessed_dir, exist_ok=True)

    n_images = 28 if split == "train" else 15
    for image_id in tqdm(range(1, n_images), desc="Preprocessing inputs"):
        output_path = os.path.join(preprocessed_dir, f"{image_id}.h5")
        if os.path.exists(output_path):
            continue

        raw_paths = natsorted(glob(os.path.join(data_dir, "tiles", f"{split}_{image_id}_*.png")))
        label_paths = [p.replace("tiles", "labels").replace(".png", ".mat") for p in raw_paths]
        _preprocess_image(raw_paths, label_paths, output_path)


def get_consep_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CoNSeP dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded and preprocessed.
    """
    data_dir = os.path.join(path, "data", "consep")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name="rftexas/tiled-consep-224x224px", download=download)
    util.unzip(zip_path=os.path.join(
        path, "tiled-consep-224x224px.zip"), dst=os.path.join(path, "data"), remove=False
    )

    return data_dir


def get_consep_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> List[str]:
    """Get paths to the CoNSeP data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    data_dir = get_consep_data(path, download)

    _preprocess_data(data_dir, "train")
    _preprocess_data(data_dir, "test")

    if split not in ["train", "test"]:
        raise ValueError(f"'{split}' is not a valid split.")

    paths = natsorted(glob(os.path.join(data_dir, "preprocessed", split, "*.h5")))
    return paths


def get_consep_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CoNSeP dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_consep_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        is_seg_dataset=True,
        patch_shape=patch_shape,
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_consep_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CoNSeP dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_consep_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
