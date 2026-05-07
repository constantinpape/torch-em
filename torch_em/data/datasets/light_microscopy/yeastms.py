"""The YeastMS dataset contains annotations for yeast cell instance segmentation
in brightfield microscopy images of microfluidic trap structures.

The dataset provides 493 annotated images (256x256) with instance segmentation
masks for both cells and trap microstructures across train/val/test splits.

The dataset is located at https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/3799.
This dataset is from the publication https://doi.org/10.48550/arXiv.2304.07597.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Union, Tuple, List, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/3799/yeast_cell_in_microstructures_dataset.zip"
CHECKSUM = "80d9e34266895a030b5dfbb81c25f9bd41e7d8c3d57f2c5aaeafd7c7c3a2d6b5"

VALID_SPLITS = ["train", "val", "test"]


def _create_h5_data(path, split):
    """Create h5 files with raw images and cell instance labels from .pt tensors."""
    import h5py
    import torch
    from natsort import natsorted
    from tqdm import tqdm

    h5_dir = os.path.join(path, "h5_data", split)
    os.makedirs(h5_dir, exist_ok=True)

    input_dir = os.path.join(path, split, "inputs")
    instance_dir = os.path.join(path, split, "instances")
    class_dir = os.path.join(path, split, "classes")

    input_paths = natsorted(glob(os.path.join(input_dir, "*.pt")))

    for input_path in tqdm(input_paths, desc=f"Creating h5 files for '{split}'"):
        fname = os.path.basename(input_path).replace(".pt", ".h5")
        h5_path = os.path.join(h5_dir, fname)

        if os.path.exists(h5_path):
            continue

        sample_id = os.path.basename(input_path)
        instance_path = os.path.join(instance_dir, sample_id)
        class_path = os.path.join(class_dir, sample_id)

        raw = torch.load(input_path, weights_only=False).numpy()
        instances = torch.load(instance_path, weights_only=False).numpy()  # (N, H, W)
        classes = torch.load(class_path, weights_only=False).numpy()  # (N,)

        # Create cell instance labels (class 0 = cell, class 1 = trap).
        labels = np.zeros(raw.shape, dtype="int64")
        cell_id = 1
        for i in range(instances.shape[0]):
            if classes[i] == 0:  # cell
                labels[instances[i] > 0] = cell_id
                cell_id += 1

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")

    return h5_dir


def get_yeastms_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the YeastMS dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the directory with the data.
    """
    data_dir = os.path.join(path, "train")
    if os.path.exists(data_dir):
        return path

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "yeast_cell_in_microstructures_dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return path


def get_yeastms_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
) -> List[str]:
    """Get paths to the YeastMS data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. One of 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the h5 data.
    """
    from natsort import natsorted

    assert split in VALID_SPLITS, f"'{split}' is not a valid split. Choose from {VALID_SPLITS}."

    get_yeastms_data(path, download)

    h5_dir = os.path.join(path, "h5_data", split)
    if not os.path.exists(h5_dir) or len(glob(os.path.join(h5_dir, "*.h5"))) == 0:
        _create_h5_data(path, split)

    h5_paths = natsorted(glob(os.path.join(h5_dir, "*.h5")))
    assert len(h5_paths) > 0, f"No data found for split '{split}'"

    return h5_paths


def get_yeastms_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the YeastMS dataset for yeast cell segmentation in microstructures.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    h5_paths = get_yeastms_paths(path, split, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True,
    )
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=h5_paths,
        raw_key="raw",
        label_paths=h5_paths,
        label_key="labels",
        patch_shape=patch_shape,
        ndim=2,
        **kwargs
    )


def get_yeastms_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the YeastMS dataloader for yeast cell segmentation in microstructures.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_yeastms_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
