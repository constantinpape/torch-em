import os
from tqdm import tqdm
from glob import glob

import z5py
import numpy as np
import pandas as pd

import torch_em

from . import util


# Automatic download is currently not possible, because of authentication
URL = None  # TODO: here - https://datasets.deepcell.org/data


def _create_split(path, split):
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
    path, split, patch_shape, download=False, **kwargs
):
    """Dataset for the segmentation of cell nuclei imaged with fluorescene microscopy.

    This dataset is from the publication https://doi.org/10.1101/803205.
    Please cite it if you use this dataset for a publication."""
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
    path, split, patch_shape, batch_size, download=False, **kwargs
):
    """Dataloader for the segmentation of cell nuclei for 5 different cell lines in fluorescence microscopes.
    See `get_dynamicnuclearnet_dataset` for details.
"""
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_dynamicnuclearnet_dataset(path, split, patch_shape, download, **ds_kwargs)
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
