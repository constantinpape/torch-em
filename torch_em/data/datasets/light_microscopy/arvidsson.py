"""This dataset contains annotations for nucleus segmentation in
high-content fluorescence microscopy images.

The dataset is located at https://zenodo.org/records/6657260.
This dataset is from the publication https://doi.org/10.1016/j.dib.2022.108769.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components

import torch_em

from torch.utils.data import Dataset, DataLoader

from .. import util


URLS = {
    "train": "https://zenodo.org/records/6657260/files/training_nuclei.zip",
    "val": "https://zenodo.org/records/6657260/files/development_nuclei.zip",
    "test": "https://zenodo.org/records/6657260/files/test_nuclei.zip",
}

CHECKSUMS = {
    "train": "df075941f4e561f9ef82d4c48d22cf97e3627a0b63fa136675197614813fff90",
    "val": "722530a93fd5b67f61d52964651c715be6227c1c0508c4c95ef2b04b52fc1dd1",
    "test": "377dc719c4eaf9bfa30273f7e3a4042d98dbbfc4a1c4af2a467879237bff592f",
}


def get_arvidsson_data(
    path: Union[os.PathLike, str], split: Literal['train', 'val', 'test'], download: bool = False
) -> str:
    """Download the Arvidsson dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the training data.
    """
    if split == "train":
        dname = "training_nuclei"
    elif split == "val":
        dname = "development_nuclei"
    elif split == "test":
        dname = "test_nuclei"
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    data_dir = os.path.join(path, dname)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"{dname}.zip")
    util.download_source(path=zip_path, url=URLS[split], download=download, checksum=CHECKSUMS[split])
    util.unzip(zip_path=os.path.join(path, f"{dname}.zip"), dst=path)

    return data_dir


def get_arvidsson_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'val', 'test'], download: bool = False,
) -> Tuple[List[int], List[int]]:
    """Get paths to the Arvidsson data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_arvidsson_data(path, split, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "images", "*.png")))
    label_paths = natsorted(glob(os.path.join(data_dir, "annotations", "*_preprocessed.tif")))
    if len(raw_paths) == len(label_paths):
        return raw_paths, label_paths

    channel_label_paths = natsorted(glob(os.path.join(data_dir, "annotations", "*.png")))
    instance_paths = []
    for rpath, lpath in tqdm(
        zip(raw_paths, channel_label_paths), desc=f"Preprocessing labels for '{split}' split", total=len(raw_paths)
    ):
        instance_path = lpath.replace(".png", "_preprocessed.tif")
        instance_paths.append(instance_path)
        if os.path.exists(instance_path):
            continue

        raw = imageio.imread(rpath)
        labels = imageio.imread(lpath)

        # NOTE: Converting the RGB-style instance labels to single channel instance labels.
        # We do not operate over the backgroun region (with known pixel values: [0, 0, 0])
        background_mask = np.all(labels == [0, 0, 0], axis=-1)
        _, indices = np.unique(labels[~background_mask].reshape(-1, 3), axis=0, return_inverse=True)

        instances = np.zeros(labels.shape[:2], dtype=np.int32)
        instances[~background_mask] = indices + 1
        instances = connected_components(instances)

        assert raw.shape == instances.shape

        imageio.imwrite(instance_path, instances, compression="zlib")

    return raw_paths, instance_paths


def get_arvidsson_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Arvidsson dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_arvidsson_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_arvidsson_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Arvidsson dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_arvidsson_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
