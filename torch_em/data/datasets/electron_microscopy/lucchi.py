"""The Lucchi dataset is a segmentation dataset for mitochondrion segmentation in electron microscopy.

The dataset was published in https://doi.org/10.48550/arXiv.1812.06024.
Please cite this publication if you use the dataset in your research.
We use the version of the dataset from https://sites.google.com/view/connectomics/.
"""

import os
from glob import glob
from tqdm import tqdm
from shutil import rmtree
from concurrent import futures
from typing import Tuple, Union, Literal

import imageio
import numpy as np

import torch_em

from torch.utils.data import Dataset, DataLoader

from .. import util


URL = "http://www.casser.io/files/lucchi_pp.zip"
CHECKSUM = "770ce9e98fc6f29c1b1a250c637e6c5125f2b5f1260e5a7687b55a79e2e8844d"


def _load_volume(path, pattern):
    nz = len(glob(os.path.join(path, "*.png")))
    im0 = imageio.imread(os.path.join(path, pattern % 0))
    out = np.zeros((nz,) + im0.shape, dtype=im0.dtype)
    out[0] = im0

    def _loadz(z):
        im = imageio.imread(os.path.join(path, pattern % z))
        out[z] = im

    n_threads = 8
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(_loadz, range(1, nz)), desc="Load volume", total=nz-1
        ))

    return out


def _create_data(root, inputs, out_path):
    import h5py

    raw = _load_volume(os.path.join(root, inputs[0]), pattern="mask%04i.png")
    labels_argb = _load_volume(os.path.join(root, inputs[1]), pattern="%i.png")
    if labels_argb.ndim == 4:
        labels = np.zeros(raw.shape, dtype="uint8")
        fg_mask = (labels_argb == np.array([255, 255, 255, 255])[None, None, None]).all(axis=-1)
        labels[fg_mask] = 1
    else:
        assert labels_argb.ndim == 3
        labels = labels_argb
        labels[labels == 255] = 1
    assert (np.unique(labels) == np.array([0, 1])).all()
    assert raw.shape == labels.shape, f"{raw.shape}, {labels.shape}"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels.astype("uint8"), compression="gzip")


def get_lucchi_data(path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False) -> str:
    """Download the Lucchi dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to download, either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath for the downloaded data.
    """
    data_path = os.path.join(path, f"lucchi_{split}.h5")
    if os.path.exists(data_path):
        return data_path

    os.makedirs(path)
    tmp_path = os.path.join(path, "lucchi.zip")
    util.download_source(tmp_path, URL, download, checksum=CHECKSUM)
    util.unzip(tmp_path, path, remove=True)

    root = os.path.join(path, "Lucchi++")
    assert os.path.exists(root), root

    inputs = [["Test_In", "Test_Out"], ["Train_In", "Train_Out"]]
    outputs = ["lucchi_train.h5", "lucchi_test.h5"]
    for inp, out in zip(inputs, outputs):
        out_path = os.path.join(path, out)
        _create_data(root, inp, out_path)
    rmtree(root)

    assert os.path.exists(data_path), data_path
    return data_path


def get_lucchi_paths(path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False) -> str:
    """Get paths to the Lucchi data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath for the stored data.
    """
    get_lucchi_data(path, split, download)
    data_path = os.path.join(path, f"lucchi_{split}.h5")
    return data_path


def get_lucchi_dataset(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    patch_shape: Tuple[int, int, int],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get dataset for EM mitochondrion segmentation in the Lucchi dataset.

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

    data_path = get_lucchi_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=data_path,
        raw_key="raw",
        label_paths=data_path,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs
    )


def get_lucchi_loader(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get dataloader for EM mitochondrion segmentation in the Lucchi dataset.

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
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lucchi_dataset(path, split, patch_shape, download=download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
