"""The Kasthuri dataset is a segmentation dataset for mitochondrion segmentation in electron microscopy.

The dataset was published in https://doi.org/10.48550/arXiv.1812.06024.
Please cite this publication if you use the dataset in your research.
We use the version of the dataset from https://sites.google.com/view/connectomics/.
"""

import os
from glob import glob
from tqdm import tqdm
from shutil import rmtree
from concurrent import futures
from typing import Tuple, Union

import imageio
import numpy as np

import torch_em

from torch.utils.data import Dataset, DataLoader

from .. import util

URL = "http://www.casser.io/files/kasthuri_pp.zip "
CHECKSUM = "bbb78fd205ec9b57feb8f93ebbdf1666261cbc3e0305e7f11583ab5157a3d792"

# TODO: add sampler for foreground (-1 is empty area)
# TODO: and masking for the empty space


def _load_volume(path):
    files = glob(os.path.join(path, "*.png"))
    files.sort()
    nz = len(files)

    im0 = imageio.imread(files[0])
    out = np.zeros((nz,) + im0.shape, dtype=im0.dtype)
    out[0] = im0

    def _loadz(z):
        im = imageio.imread(files[z])
        out[z] = im

    n_threads = 8
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(_loadz, range(1, nz)), desc="Load volume", total=nz-1
        ))

    return out


def _create_data(root, inputs, out_path):
    import h5py

    raw = _load_volume(os.path.join(root, inputs[0]))
    labels_argb = _load_volume(os.path.join(root, inputs[1]))
    assert labels_argb.ndim == 4
    labels = np.zeros(raw.shape, dtype="int8")

    fg_mask = (labels_argb == np.array([255, 255, 255])[None, None, None]).all(axis=-1)
    labels[fg_mask] = 1
    bg_mask = (labels_argb == np.array([2, 2, 2])[None, None, None]).all(axis=-1)
    labels[bg_mask] = -1
    assert (np.unique(labels) == np.array([-1, 0, 1])).all()
    assert raw.shape == labels.shape, f"{raw.shape}, {labels.shape}"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")


def get_kasthuri_data(path: Union[os.PathLike, str], download: bool) -> str:
    """Download the kasthuri dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath for the downloaded data.
    """
    if os.path.exists(path):
        return path

    os.makedirs(path)
    tmp_path = os.path.join(path, "kasthuri.zip")
    util.download_source(tmp_path, URL, download, checksum=CHECKSUM)
    util.unzip(tmp_path, path, remove=True)

    root = os.path.join(path, "Kasthuri++")
    assert os.path.exists(root), root

    inputs = [["Test_In", "Test_Out"], ["Train_In", "Train_Out"]]
    outputs = ["kasthuri_train.h5", "kasthuri_test.h5"]
    for inp, out in zip(inputs, outputs):
        out_path = os.path.join(path, out)
        _create_data(root, inp, out_path)

    rmtree(root)
    return path


def get_kasthuri_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int, int],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get dataset for EM mitochondrion segmentation in the kasthuri dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train' or 'test'.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert split in ("train", "test")
    get_kasthuri_data(path, download)
    data_path = os.path.join(path, f"kasthuri_{split}.h5")
    assert os.path.exists(data_path), data_path
    raw_key, label_key = "raw", "labels"
    return torch_em.default_segmentation_dataset(data_path, raw_key, data_path, label_key, patch_shape, **kwargs)


def get_kasthuri_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get dataloader for EM mitochondrion segmentation in the kasthuri dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train' or 'test'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The PyTorch DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_kasthuri_dataset(path, split, patch_shape, download=download, **ds_kwargs)
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
