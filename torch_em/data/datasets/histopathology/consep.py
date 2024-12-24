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

import numpy as np
from scipy.io import loadmat
import imageio.v3 as imageio
from skimage.measure import label as connected_components

from torch.utils.data import Dataset, DataLoader

import torch_em

from elf.segmentation.stitching import stitch_tiled_segmentation

from .. import util


def _preprocess_data(data_dir, split):
    import h5py

    preprocessed_dir = os.path.join(data_dir, "preprocessed", split)
    os.makedirs(preprocessed_dir, exist_ok=True)
    for i in tqdm(range(1, 28 if split == "train" else 15), desc="Preprocessing inputs"):
        raw_paths = natsorted(glob(os.path.join(data_dir, "tiles", f"{split}_{i}_*.png")))
        label_paths = [p.replace("tiles", "labels").replace(".png", ".mat") for p in raw_paths]

        raw_tiles, label_tiles, tile_shapes = [], [], []
        for rpath, lpath in zip(raw_paths, label_paths):
            tile_shapes.append(tuple(int(t) for t in Path(rpath).stem.split("_")[2:]))
            raw_tiles.append(imageio.imread(rpath))
            label_tiles.append(loadmat(lpath)["instance_map"])

        h = max(shape[1] for shape in tile_shapes)
        w = max(shape[3] for shape in tile_shapes)

        labels = np.zeros((h, w))

        raw = np.zeros((h, w, 3))
        for tile, shape in zip(raw_tiles, tile_shapes):
            y1, y2, x1, x2 = shape
            raw[y1: y2, x1: x2] = tile

        labels = np.zeros((h, w))
        offset = 0
        for tile, shape in zip(label_tiles, tile_shapes):
            # We need to make sure that each object labels are distinct across all tiles.
            # We separate them by running connected components and setting a recurring offset to all instances per tile.
            tile = connected_components(tile).astype("uint32")
            tile[tile != 0] += offset
            offset = int(tile.max())

            y1, y2, x1, x2 = shape
            labels[y1: y2, x1: x2] = tile

        stitched_labels = stitch_tiled_segmentation(labels, tile_shape=(224, 224), overlap=1)

        import napari
        v = napari.Viewer()
        v.add_labels(labels.astype("uint32"))
        v.add_labels(stitched_labels)
        napari.run()

        volume_path = os.path.join(preprocessed_dir, f"{i}.h5")
        with h5py.File(volume_path, "w") as f:
            f.create_dataset("raw", data=raw.transpose(2, 0, 1), compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


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
    path: Union[os.PathLike, str], split: Literal['train', 'test'], download: bool = False
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

    if split not in ['train', 'test']:
        raise ValueError(f"'{split}' is not a valid split.")

    volume_paths = natsorted(glob(os.path.join(data_dir, "preprocessed", split, "*.h5")))
    return volume_paths


def get_consep_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'],
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
    split: Literal['train', 'test'],
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
