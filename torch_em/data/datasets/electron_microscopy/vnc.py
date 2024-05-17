"""The VNC dataset contains segmentation annotations for mitochondria in EM.

It contains two volumes from TEM of the drosophila brain.
Please cite https://doi.org/10.6084/m9.figshare.856713.v1 if you use this dataset in your publication.
"""

import os
from glob import glob
from shutil import rmtree
from typing import List, Optional, Union, Tuple

import imageio
import h5py
import numpy as np
import torch_em
from skimage.measure import label
from torch.utils.data import Dataset, DataLoader

from .. import util

URL = "https://github.com/unidesigner/groundtruth-drosophila-vnc/archive/refs/heads/master.zip"
CHECKSUM = "f7bd0db03c86b64440a16b60360ad60c0a4411f89e2c021c7ee2c8d6af3d7e86"


def _create_volume(f, key, pattern, process=None):
    images = glob(pattern)
    images.sort()
    data = np.concatenate([imageio.imread(im)[None] for im in images], axis=0)
    if process is not None:
        data = process(data)
    f.create_dataset(key, data=data, compression="gzip")


def get_vnc_data(path: Union[os.PathLike, str], download: bool) -> str:
    """Download the VNC training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The path to the downloaded data.
    """

    train_path = os.path.join(path, "vnc_train.h5")
    test_path = os.path.join(path, "vnc_test.h5")
    if os.path.exists(train_path) and os.path.exists(test_path):
        return path

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "vnc.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    util.unzip(zip_path, path, remove=True)

    root = os.path.join(path, "groundtruth-drosophila-vnc-master")
    assert os.path.exists(root)

    with h5py.File(train_path, "w") as f:
        _create_volume(f, "raw", os.path.join(root, "stack1", "raw", "*.tif"))
        _create_volume(f, "labels/mitochondria", os.path.join(root, "stack1", "mitochondria", "*.png"), process=label)
        _create_volume(f, "labels/synapses", os.path.join(root, "stack1", "synapses", "*.png"), process=label)
        # TODO find the post-processing to go from neuron labels to membrane labels
        # _create_volume(f, "labels/neurons", os.path.join(root, "stack1", "membranes", "*.png"))

    with h5py.File(test_path, "w") as f:
        _create_volume(f, "raw", os.path.join(root, "stack2", "raw", "*.tif"))

    rmtree(root)
    return path


def get_vnc_mito_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the VNC dataset for segmentating mitochondria in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    get_vnc_data(path, download)
    data_path = os.path.join(path, "vnc_train.h5")

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, boundaries=boundaries, offsets=offsets, binary=binary,
    )

    raw_key = "raw"
    label_key = "labels/mitochondria"
    return torch_em.default_segmentation_dataset(data_path, raw_key, data_path, label_key, patch_shape, **kwargs)


def get_vnc_mito_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the VNC dataloader for segmentating mitochondria in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
       The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_vnc_mito_dataset(
        path, patch_shape, download=download, offsets=offsets, boundaries=boundaries, binary=binary, **kwargs
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)


# TODO implement
def get_vnc_neuron_loader(path, patch_shape, download=False, **kwargs):
    raise NotImplementedError
