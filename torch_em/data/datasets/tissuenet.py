import os
from glob import glob

import z5py
import numpy as np
import torch_em
from tqdm import tqdm

from .util import unzip


# Automated download is currently not possible, because of authentication
URL = None  # TODO: here - https://datasets.deepcell.org/data


def _create_split(path, split):
    split_file = os.path.join(path, f"tissuenet_v1.1_{split}.npz")
    split_folder = os.path.join(path, split)
    os.makedirs(split_folder, exist_ok=True)
    data = np.load(split_file)
    x, y = data["X"], data["y"]
    for i, (im, label) in tqdm(enumerate(zip(x, y)), total=len(x), desc=f"Creating files for {split}-split"):
        out_path = os.path.join(split_folder, f"image_{i:04}.n5")
        with z5py.File(out_path, "a") as f:
            f.create_dataset("raw/nucleus", data=im[..., 0], compression="gzip", chunks=im[..., 0].shape)
            f.create_dataset("raw/cell", data=im[..., 1], compression="gzip", chunks=im[..., 1].shape)
            # the swithh 0<->1 is intentional, the data format is chaotic...
            f.create_dataset("labels/nucleus", data=label[..., 1], compression="gzip", chunks=label[..., 1].shape)
            f.create_dataset("labels/cell", data=label[..., 0], compression="gzip", chunks=label[..., 0].shape)
    os.remove(split_file)


def _create_dataset(path, zip_path):
    unzip(zip_path, path, remove=False)
    splits = ["train", "val", "test"]
    assert all([os.path.exists(os.path.join(path, f"tissuenet_v1.1_{split}.npz")) for split in splits])
    for split in splits:
        _create_split(path, split)


def get_tissuenet_loader(path, split, mode, download=False, **kwargs):
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
    data_path = glob(os.path.join(split_folder, "*.n5"))
    assert len(data_path) > 0
    print(len(data_path))

    assert mode in ["nucleus", "cell"], f"Got {mode}"
    raw_key, label_key = f"raw/{mode}", f"labels/{mode}"
    return torch_em.default_segmentation_loader(
        data_path, raw_key, data_path, label_key, is_seg_dataset=True, ndim=2, **kwargs
    )
