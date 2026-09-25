"""The MBAS dataset contains annotations for multi-class bi-atrial segmentation in
late gadolinium-enhanced (LGE) cardiac MRI.

The data was curated for the MBAS 2024 challenge (Multi-class Bi-Atrial Segmentation), which was held together
with the STACOM workshop at MICCAI 2024. The public release consists of 100 3D LGE-MRI, split into the 70
studies of the official training set and the 30 studies of the official additional labeled set, which is
selected with the 'split' argument. Each study comes with a multi-class segmentation mask with the label ids
described in `LABEL_IDS`: 1 = right atrial wall, 2 = left atrial wall, 3 = right atrial cavity,
4 = left atrial cavity.

The data is located at https://zenodo.org/records/19120533 and is released under a custom license: use for
non-commercial AI development and testing is permitted and citation is mandatory, while commercialization or
other uses require written permission from the authors.

This dataset is from the publication https://doi.org/10.1016/j.media.2026.104203.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "train": "https://zenodo.org/records/19120533/files/MBAS_Training_4C.zip?download=1",
    "val": "https://zenodo.org/records/19120533/files/MBAS_Testing_4C.zip?download=1",
}

CHECKSUMS = {
    "train": "a765bff61e642ac82a6aa5fffcc39755f75f753db58c111e6e2f50a88a909816",
    "val": "27c235e2646b37e55734e3a213ad24c56e87540bb0e0c245b6a935a9d6220883",
}

LABEL_IDS = {"background": 0, "raw": 1, "law": 2, "ra": 3, "la": 4}


def get_mbas_data(path: Union[os.PathLike, str], split: Literal["train", "val"], download: bool = False) -> str:
    """Download the MBAS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' (70 cases) or 'val' (30 additional cases).
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    if split not in URLS:
        raise ValueError(f"'{split}' is not a valid split. Please choose one of {list(URLS.keys())}.")

    data_dir = os.path.join(path, split)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"MBAS_{split}.zip")
    util.download_source(path=zip_path, url=URLS[split], download=download, checksum=CHECKSUMS[split])
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def get_mbas_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the MBAS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' (70 cases) or 'val' (30 additional cases).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_mbas_data(path, split, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "*", "*_image.nii.gz")))
    label_paths = natsorted(glob(os.path.join(data_dir, "*", "*_label.nii.gz")))

    if len(raw_paths) == 0 or len(raw_paths) != len(label_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return raw_paths, label_paths


def get_mbas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MBAS dataset for multi-class bi-atrial segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' (70 cases) or 'val' (30 additional cases).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_mbas_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_mbas_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MBAS dataloader for multi-class bi-atrial segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' (70 cases) or 'val' (30 additional cases).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mbas_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
