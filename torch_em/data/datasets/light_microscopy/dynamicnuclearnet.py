"""The DynamicNuclearNet dataset contains annotations for nucleus segmentation
and tracking in fluorescence light microscopy, for five different cell lines.

This dataset is from the publication https://doi.org/10.1101/803205.
Please cite it if you use this dataset for your research.

This dataset cannot be downloaded automatically, please visit https://datasets.deepcell.org/data
and download it yourself.
"""

import os
from tqdm import tqdm
from glob import glob
from typing import Tuple, Union

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def _create_split(path, split):
    import z5py

    split_file = os.path.join(path, "DynamicNuclearNet-segmentation-v1_0", f"{split}.npz")
    split_folder = os.path.join(path, split)
    os.makedirs(split_folder, exist_ok=True)
    data = np.load(split_file, allow_pickle=True)

    x, y = data["X"], data["y"]
    metadata = data["meta"]
    metadata = pd.DataFrame(metadata[1:], columns=metadata[0])

    for i, (im, label) in tqdm(enumerate(zip(x, y)), total=len(x), desc=f"Creating files for {split}-split"):
        out_path = os.path.join(split_folder, f"image_{i:04}.zarr")
        image_channel = im[..., 0]
        label_channel = label[..., 0]
        chunks = image_channel.shape
        with z5py.File(out_path, "a") as f:
            f.create_dataset("raw", data=image_channel, compression="gzip", chunks=chunks)
            f.create_dataset("labels", data=label_channel, compression="gzip", chunks=chunks)

    os.remove(split_file)


def _create_dataset(path, zip_path):
    util.unzip(zip_path, path, remove=False)
    splits = ["train", "val", "test"]
    assert all(
        [os.path.exists(os.path.join(path, "DynamicNuclearNet-segmentation-v1_0", f"{split}.npz")) for split in splits]
    )
    for split in splits:
        _create_split(path, split)


def get_dynamicnuclearnet_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DynamicNuclearNet dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use for the dataset. Either 'train', 'val' or 'test'.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    splits = ["train", "val", "test"]
    assert split in splits

    # check if the dataset exists already
    zip_path = os.path.join(path, "DynamicNuclearNet-segmentation-v1_0.zip")
    if all([os.path.exists(os.path.join(path, split)) for split in splits]):  # yes it does
        pass
    elif os.path.exists(zip_path):  # no it does not, but we have the zip there and can unpack it
        _create_dataset(path, zip_path)
    else:
        raise RuntimeError(
            "We do not support automatic download for the dynamic nuclear net dataset yet. "
            f"Please download the dataset from https://datasets.deepcell.org/data and put it here: {zip_path}"
        )

    split_folder = os.path.join(path, split)
    assert os.path.exists(split_folder)
    data_path = glob(os.path.join(split_folder, "*.zarr"))
    assert len(data_path) > 0

    raw_key, label_key = "raw", "labels"

    return torch_em.default_segmentation_dataset(
        data_path, raw_key, data_path, label_key, patch_shape, is_seg_dataset=True, ndim=2, **kwargs
    )


def get_dynamicnuclearnet_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DynamicNuclearNet dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use for the dataset. Either 'train', 'val' or 'test'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_dynamicnuclearnet_dataset(path, split, patch_shape, download, **ds_kwargs)
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
