"""The TissueNet dataset contains annotations for cell and nucleus segmentation in microscopy images
of different tissue types.

This dataset is from the publication https://doi.org/10.1038/s41587-021-01094-0.
Please cite it if you use this dataset for your research.

This dataset cannot be downloaded automatically, please visit https://datasets.deepcell.org/data
and download it yourself.
"""

import os
from glob import glob
from tqdm import tqdm
from typing import Tuple, Union, List, Literal

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def _create_split(path, split):
    import z5py

    split_file = os.path.join(path, f"tissuenet_v1.1_{split}.npz")
    split_folder = os.path.join(path, split)
    os.makedirs(split_folder, exist_ok=True)
    data = np.load(split_file, allow_pickle=True)

    x, y = data["X"], data["y"]
    metadata = data["meta"]
    metadata = pd.DataFrame(metadata[1:], columns=metadata[0])

    for i, (im, label) in tqdm(enumerate(zip(x, y)), total=len(x), desc=f"Creating files for {split}-split"):
        out_path = os.path.join(split_folder, f"image_{i:04}.zarr")
        nucleus_channel = im[..., 0]
        cell_channel = im[..., 1]
        rgb = np.stack([cell_channel, nucleus_channel, np.zeros_like(nucleus_channel)])
        chunks = cell_channel.shape
        with z5py.File(out_path, "a") as f:

            f.create_dataset("raw/nucleus", data=nucleus_channel, compression="gzip", chunks=chunks)
            f.create_dataset("raw/cell", data=cell_channel, compression="gzip", chunks=chunks)
            f.create_dataset("raw/rgb", data=rgb, compression="gzip", chunks=(3,) + chunks)

            # the switch 0<->1 is intentional, the data format is chaotic...
            f.create_dataset("labels/nucleus", data=label[..., 1], compression="gzip", chunks=chunks)
            f.create_dataset("labels/cell", data=label[..., 0], compression="gzip", chunks=chunks)

    os.remove(split_file)


def _create_dataset(path, zip_path):
    util.unzip(zip_path, path, remove=False)
    splits = ["train", "val", "test"]
    assert all([os.path.exists(os.path.join(path, f"tissuenet_v1.1_{split}.npz")) for split in splits])
    for split in splits:
        _create_split(path, split)


def get_tissuenet_data(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False
) -> str:
    """Obtain the TissueNet dataset.

    NOTE: Automatic download is not supported for TissueNet dataset.
    Please download the dataset from https://datasets.deepcell.org/data.

    Args:
        path: Filepath to a folder where the manually downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The path where inputs are stored per split.
    """
    splits = ["train", "val", "test"]
    assert split in splits

    # check if the dataset exists already
    zip_path = os.path.join(path, "tissuenet_v1.1.zip")
    if all([os.path.exists(os.path.join(path, split)) for split in splits]):  # yes it does
        pass
    elif os.path.exists(zip_path):  # no it does not, but we have the zip there and can unpack it
        _create_dataset(path, zip_path)
    else:
        raise RuntimeError(
            "We do not support automatic download for the tissuenet datasets yet."
            f"Please download the dataset from https://datasets.deepcell.org/data and put it here: {zip_path}"
        )

    split_folder = os.path.join(path, split)
    return split_folder


def get_tissuenet_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False
) -> List[str]:
    """Get paths to the TissueNet data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the data.
    """
    split_folder = get_tissuenet_data(path, split, download)
    assert os.path.exists(split_folder)
    data_paths = glob(os.path.join(split_folder, "*.zarr"))
    assert len(data_paths) > 0

    return data_paths


def get_tissuenet_dataset(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    patch_shape: Tuple[int, int],
    raw_channel: Literal["nucleus", "cell", "rgb"],
    label_channel: Literal["nucleus", "cell"],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the TissueNet dataset for segmenting cells and nucleus in microscopy tissue images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        patch_shape: The patch shape to use for training.
        raw_channel: The channel to load for the raw data. Either 'nucleus', 'cell' or 'rgb'.
        label_channel: The channel to load for the label data. Either 'nucleus' or 'cell'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert raw_channel in ("nucleus", "cell", "rgb")
    assert label_channel in ("nucleus", "cell")

    data_paths = get_tissuenet_paths(path, split, download)

    with_channels = True if raw_channel == "rgb" else False
    kwargs = util.update_kwargs(kwargs, "with_channels", with_channels)
    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs = util.update_kwargs(kwargs, "ndim", 2)

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key=f"raw/{raw_channel}",
        label_paths=data_paths,
        label_key=f"labels/{label_channel}",
        patch_shape=patch_shape,
        **kwargs
    )


# TODO enable loading specific tissue types etc. (from the 'meta' attributes)
def get_tissuenet_loader(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    patch_shape: Tuple[int, int],
    batch_size: int,
    raw_channel: Literal["nucleus", "cell", "rgb"],
    label_channel: Literal["nucleus", "cell"],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the TissueNet dataloader for segmenting cells and nucleus in microscopy tissue images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        raw_channel: The channel to load for the raw data. Either 'nucleus', 'cell' or 'rgb'.
        label_channel: The channel to load for the label data. Either 'nucleus' or 'cell'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_tissuenet_dataset(path, split, patch_shape, raw_channel, label_channel, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
