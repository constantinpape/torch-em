import os
from glob import glob

import numpy as np
import pandas as pd
import torch_em
import z5py

from tqdm import tqdm
from .. import util


# Automated download is currently not possible, because of authentication
URL = None  # TODO: here - https://datasets.deepcell.org/data


def _create_split(path, split):
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
        rgb = np.stack([np.zeros_like(nucleus_channel), cell_channel, nucleus_channel])
        chunks = cell_channel.shape
        with z5py.File(out_path, "a") as f:

            f.create_dataset("raw/nucleus", data=im[..., 0], compression="gzip", chunks=chunks)
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


def get_tissuenet_dataset(
    path, split, patch_shape, raw_channel, label_channel, download=False, **kwargs
):
    """Dataset for the segmentation of cells in tissue imaged with light microscopy.

    This dataset is from the publication https://doi.org/10.1038/s41587-021-01094-0.
    Please cite it if you use this dataset for a publication.
    """
    assert raw_channel in ("nucleus", "cell", "rgb")
    assert label_channel in ("nucleus", "cell")

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
    assert os.path.exists(split_folder)
    data_path = glob(os.path.join(split_folder, "*.zarr"))
    assert len(data_path) > 0

    raw_key, label_key = f"raw/{raw_channel}", f"labels/{label_channel}"

    with_channels = True if raw_channel == "rgb" else False
    kwargs = util.update_kwargs(kwargs, "with_channels", with_channels)
    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs = util.update_kwargs(kwargs, "ndim", 2)

    return torch_em.default_segmentation_dataset(data_path, raw_key, data_path, label_key, patch_shape, **kwargs)


# TODO enable loading specific tissue types etc. (from the 'meta' attributes)
def get_tissuenet_loader(
    path, split, patch_shape, batch_size, raw_channel, label_channel, download=False, **kwargs
):
    """Dataloader for the segmentation of cells in tissue imaged with light microscopy.
    See 'get_tissuenet_dataset' for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_tissuenet_dataset(
        path, split, patch_shape, raw_channel, label_channel, download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
