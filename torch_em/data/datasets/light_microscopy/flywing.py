"""The FlyWing dataset contains annotated fluorescence microscopy images of Drosophila wing epithelia.

This loader uses the zero-noise segmentation release published with DenoiSeg. It contains 1428 training and
252 validation patches of shape 128 x 128, as well as 42 test images of shape 512 x 512. The data is derived from
the epithelial cell tracking benchmark introduced in https://doi.org/10.1007/978-3-030-11024-6_33.

The dataset is located at https://doi.org/10.5281/zenodo.5156991 under the CC BY 4.0 license. This release does
not contain the eight complete time-lapse movies used by the original tracking benchmark or its movie-level splits.
It is from the publication https://doi.org/10.1007/978-3-030-66415-2_21. Please cite the dataset and publications
if you use this dataset in your research.
"""

import os
from glob import glob
from typing import List, Literal, Optional, Tuple, Union

import imageio.v3 as imageio
import numpy as np
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/5156991/files/Flywing_n0.zip/content"
CHECKSUM = "3fb49ba44e7e3e20b4fc3c77754f1bbff7184af7f343f23653f258d50e5d5aca"

SPLIT_INFO = {
    "train": ("train/train_data.npz", "X_train", "Y_train", 1428),
    "val": ("train/train_data.npz", "X_val", "Y_val", 252),
    "test": ("test/test_data.npz", "X_test", "Y_test", 42),
}


def _get_split_paths(data_dir: str, split: str) -> Tuple[List[str], List[str]]:
    raw_paths = natsorted(glob(os.path.join(data_dir, split, "images", "*.tif")))
    label_paths = natsorted(glob(os.path.join(data_dir, split, "labels", "*.tif")))
    return raw_paths, label_paths


def _is_complete(data_dir: str, split: str) -> bool:
    raw_paths, label_paths = _get_split_paths(data_dir, split)
    expected_images = SPLIT_INFO[split][3]
    return (
        len(raw_paths) == expected_images
        and len(label_paths) == expected_images
        and [os.path.basename(path) for path in raw_paths] == [os.path.basename(path) for path in label_paths]
    )


def _preprocess_split(data_dir: str, split: str) -> None:
    npz_name, raw_key, label_key, expected_images = SPLIT_INFO[split]
    npz_path = os.path.join(data_dir, npz_name)
    if not os.path.exists(npz_path):
        raise RuntimeError(f"Could not find the FlyWing source data at '{npz_path}'.")

    raw_dir = os.path.join(data_dir, split, "images")
    label_dir = os.path.join(data_dir, split, "labels")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    with np.load(npz_path) as data:
        raw = data[raw_key]
        labels = data[label_key]
        if raw.shape != labels.shape or len(raw) != expected_images:
            raise RuntimeError(
                f"Unexpected FlyWing data for split '{split}': raw={raw.shape}, labels={labels.shape}."
            )

        for index, (image, instances) in tqdm(
            enumerate(zip(raw, labels)), total=expected_images, desc=f"Preprocessing FlyWing '{split}' split"
        ):
            filename = f"image_{index:04d}.tif"
            raw_path = os.path.join(raw_dir, filename)
            label_path = os.path.join(label_dir, filename)
            if not os.path.exists(raw_path):
                imageio.imwrite(raw_path, image, compression="zlib")
            if not os.path.exists(label_path):
                imageio.imwrite(label_path, instances, compression="zlib")


def get_flywing_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download and preprocess the FlyWing segmentation dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the preprocessed data.
    """
    data_dir = os.path.join(path, "Flywing_n0")
    if all(_is_complete(data_dir, split) for split in SPLIT_INFO):
        return data_dir

    os.makedirs(path, exist_ok=True)
    if not all(os.path.exists(os.path.join(data_dir, info[0])) for info in SPLIT_INFO.values()):
        zip_path = os.path.join(path, "Flywing_n0.zip")
        util.download_source(zip_path, URL, download, CHECKSUM)
        util.unzip(zip_path, path)

    for split in SPLIT_INFO:
        if not _is_complete(data_dir, split):
            _preprocess_split(data_dir, split)

    incomplete_splits = [split for split in SPLIT_INFO if not _is_complete(data_dir, split)]
    if incomplete_splits:
        raise RuntimeError(f"FlyWing preprocessing failed for splits: {incomplete_splits}.")
    return data_dir


def get_flywing_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the FlyWing fluorescence images and cell instance labels.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The image paths and corresponding label paths.
    """
    if split not in SPLIT_INFO:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLIT_INFO)}.")

    data_dir = get_flywing_data(path, download)
    raw_paths, label_paths = _get_split_paths(data_dir, split)
    return raw_paths, label_paths


def get_flywing_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the FlyWing dataset for epithelial cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'train', 'val' or 'test'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The FlyWing patch shape must be two-dimensional, got {patch_shape}.")

    raw_paths, label_paths = get_flywing_paths(path, split, download)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary,
    )
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs,
    )


def get_flywing_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the FlyWing dataloader for epithelial cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'train', 'val' or 'test'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_flywing_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
