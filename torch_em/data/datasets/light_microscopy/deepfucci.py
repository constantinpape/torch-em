"""The DeepFUCCI dataset contains annotations for nucleus instance segmentation and cell cycle classification
in multiplexed FUCCI (fluorescent ubiquitination-based cell cycle indicator) fluorescence microscopy images.

The training data of the DeepFUCCI publication (the 'fucciphase' Zenodo record) consists of 513 images with 3 channels
(float values, normalized per image and not in [0, 1]) and sizes between 256x256 and 500x500 pixels, from several
cell lines and acquisitions (e.g. HaCaT cells at 100x, a scratch assay, 20x and 40x recordings). Each image has an
instance mask following the StarDist convention (0 is background, every nucleus has its own id) and a json file that
maps every instance id to one of the 3 cell cycle classes 1, 2 and 3. The authors define the split of the images
('training': 436 and 'validation': 77 images) in 'dataset_split.json'. The order of the 3 channels and the meaning of
the class ids are not documented in the record, so they are not interpreted by this module.

The record also contains a small independent test set (data_set_HT1080.zip), custom trained StarDist, InstanSeg and
Cellpose-SAM models and analysis data, which are not used here.

The data is located at https://doi.org/10.5281/zenodo.19671003, released under a CC-BY-4.0 license.
Please cite the corresponding Zenodo record if you use this dataset in your research.
"""

import os
import json
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/19671003/files/training_data.zip/content"
CHECKSUM = "03be8d4caf38cd7780ea2043a32697ef63defd25468f4953a8844890545f668b"

SPLITS = {"train": "training", "val": "validation"}


def get_deepfucci_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the DeepFUCCI training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data.
    """
    data_dir = os.path.join(path, "training_data")
    if os.path.exists(os.path.join(data_dir, "dataset_split.json")):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "training_data.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path, remove=False)

    assert os.path.exists(data_dir), f"The extraction of the archive did not create '{data_dir}'."
    return data_dir


def get_deepfucci_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val"] = "train", download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the DeepFUCCI training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose one of {list(SPLITS)}.")

    data_dir = get_deepfucci_data(path, download)
    with open(os.path.join(data_dir, "dataset_split.json")) as f:
        names = natsorted(json.load(f)[SPLITS[split]])

    raw_paths = [os.path.join(data_dir, "images", name) for name in names]
    label_paths = [os.path.join(data_dir, "masks", name) for name in names]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in raw_paths + label_paths)

    return raw_paths, label_paths


def get_deepfucci_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"] = "train",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DeepFUCCI dataset for nucleus instance segmentation in multiplexed FUCCI microscopy images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training. The smallest images have a size of 256x256 pixels.
        split: The choice of data split. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_deepfucci_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_deepfucci_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"] = "train",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DeepFUCCI dataloader for nucleus instance segmentation in multiplexed FUCCI microscopy images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training. The smallest images have a size of 256x256 pixels.
        split: The choice of data split. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_deepfucci_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
