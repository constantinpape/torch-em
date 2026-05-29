"""ProbTEM dataset for mitochondria segmentation in 2D TEM images.

The dataset contains TEM images of skeletal muscle with binary semantic segmentation
masks for mitochondria (0=background, 1=mitochondria). Images are 2560 x 2560 pixels
at 65 nm sample thickness acquired with a JEM-1011 microscope at 80 kV.

The dataset has 21 training and 6 test images. There is no validation split.

Masks are stored as grayscale PNGs (0=background, 255=mitochondria) with slight
anti-aliased edges. They are thresholded to binary during preprocessing.

This dataset is from the publication https://doi.org/10.1038/s41598-025-03311-1.
Please cite it if you use this dataset in your research.

The data is available at https://yoonlab.unist.ac.kr/index.php/research/mitochondria-tem-dataset/
and requires a Google Drive download via gdown: pip install gdown.
"""

import os
from glob import glob
from typing import List, Literal, Tuple, Union

import h5py
import imageio.v3 as imageio
import numpy as np

import torch_em
from torch.utils.data import Dataset, DataLoader
from .. import util


PROBTEM_GDRIVE_FOLDER = "1n2ZqbJEHPyMB_6a6OTBBACt5Jct2PZJc"
PROBTEM_DATA_ROOT = "Deeppi-EM/mitoseg_deploy/datasets/Skeletal_muscle"


def _preprocess_probtem(raw_dir, label_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw_paths = sorted(glob(os.path.join(raw_dir, "*.tif")) + glob(os.path.join(raw_dir, "*.tiff")))
    for rp in raw_paths:
        name = os.path.splitext(os.path.basename(rp))[0]
        out_path = os.path.join(out_dir, f"{name}.h5")
        if os.path.exists(out_path):
            continue

        raw = imageio.imread(rp)
        if raw.ndim == 3:
            raw = raw[..., 0]

        label_name = name.replace("x_", "y_")
        lp = os.path.join(label_dir, f"{label_name}.png")
        if not os.path.exists(lp):
            continue

        labels = imageio.imread(lp)
        if labels.ndim == 3:
            labels = labels[..., 0]
        labels = (labels >= 127).astype(np.uint8)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_probtem_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"] = "train",
    download: bool = False,
) -> str:
    """Download and preprocess the ProbTEM dataset.

    Args:
        path: Filepath to a folder where the data will be saved.
        split: The data split to use, either "train" or "test".
        download: Whether to download the data if not present.

    Returns:
        Path to the folder containing preprocessed HDF5 files.
    """
    processed_dir = os.path.join(str(path), "processed", split)
    if os.path.isdir(processed_dir) and len(glob(os.path.join(processed_dir, "*.h5"))) > 0:
        return processed_dir

    raw_dir = os.path.join(str(path), PROBTEM_DATA_ROOT, split, "input")
    label_dir = os.path.join(str(path), PROBTEM_DATA_ROOT, split, "target")

    if not os.path.isdir(raw_dir):
        if not download:
            raise RuntimeError(
                f"ProbTEM data not found at '{path}'. Set download=True or download manually from "
                "https://yoonlab.unist.ac.kr/index.php/research/mitochondria-tem-dataset/ "
                "and place in the given path."
            )
        try:
            import gdown
        except ImportError:
            raise ImportError("gdown is required to download ProbTEM: pip install gdown")
        gdown.download_folder(id=PROBTEM_GDRIVE_FOLDER, output=str(path), quiet=False)

    _preprocess_probtem(raw_dir, label_dir, processed_dir)
    return processed_dir


def get_probtem_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"] = "train",
    download: bool = False,
) -> List[str]:
    """Get paths to ProbTEM HDF5 files.

    Args:
        path: Filepath to a folder where the data will be saved.
        split: The data split to use, either "train" or "test".
        download: Whether to download the data if not present.

    Returns:
        List of paths to HDF5 files.
    """
    processed_dir = get_probtem_data(path, split, download)
    return sorted(glob(os.path.join(processed_dir, "*.h5")))


def get_probtem_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the ProbTEM dataset for mitochondria segmentation in 2D TEM images.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape (H, W) for training.
        split: The data split to use, either "train" or "test".
        download: Whether to download the data if not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 2
    data_paths = get_probtem_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="raw",
        label_paths=data_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs,
    )


def get_probtem_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for mitochondria segmentation in the ProbTEM dataset.

    Args:
        path: Filepath to a folder where the data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape (H, W) for training.
        split: The data split to use, either "train" or "test".
        download: Whether to download the data if not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_probtem_dataset(path=path, patch_shape=patch_shape, split=split, download=download, **ds_kwargs)
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
