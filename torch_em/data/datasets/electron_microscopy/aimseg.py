"""
"""

import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Tuple, List, Union, Optional, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = [
    "https://zenodo.org/records/8351731/files/Control_Dataset.rar",
    "https://zenodo.org/records/8351731/files/Validation_Dataset_v1.rar",
]

CHECKSUMS = [
    "ecd569a5f91166a09d93d29a10e2ddd2eaa3e82df531785b7aa243e426467673",
    "647216eb09a644be8980224a52d8168fa2fa5a1fd0537fb1e5d6102ec30e396d"
]


def get_aimseg_data(path: Union[os.PathLike, str], download: bool = False):
    """Get the AimSeg data.

    Args:
        path: Path to a folder where the data is downloaded.
        download: Whether to download the data if it is not present.
    """
    for url, checksum in zip(URLS, CHECKSUMS):
        rarfname = url.rsplit("/")[-1]
        dirname = Path(rarfname).stem

        if os.path.exists(os.path.join(path, dirname)):
            continue

        os.makedirs(path, exist_ok=True)

        util.download_source(path=os.path.join(path, rarfname), url=url, download=download, checksum=checksum)
        util.unzip_rarfile(rar_path=os.path.join(path, rarfname), dst=path)


def get_aimseg_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["control", "validation"]] = None,
    targets: Literal["instances", "semantic"] = "instances",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the AimSeg data.

    Args:
        path: Filepath to a folder where the data is downloaded.
        split: The split of the data to be used for training.
            Either `control` focused on healthy control specimen,
            or `validation` focused on mice undergoing remyelination.
        targets: The choice of support labels for the task.
            Either `instances` for annotated myelinated axons or `semantic` for axons, inner tongue and myelins.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    # Download the AimSeg data.
    get_aimseg_data(path, download)

    # Get the directory name for desired targets.
    if targets == "instances":
        dirname = "GroundTruth_Instance"
    elif targets == "semantic":
        dirname = "GroundTruth_Semantic"
    else:
        raise ValueError(f"'{targets}' is not a valid target choice. Please choose from 'instances' / 'semantic'.")

    # Get the paths to image and corresponding labels
    raw_paths, label_paths = [], []
    if split and split not in ["control", "validation"]:
        raise ValueError(f"'{split}' is not a valid split choice. Please choose from 'control' / 'validation'.")

    if split != "validation":
        raw_paths.extend(natsorted(glob(os.path.join(path, "Control_Dataset", "Images", "*.tif"))))
        label_paths.extend(natsorted(glob(os.path.join(path, "Control_Dataset", dirname, "*.tif"))))

    if split != "control":
        raw_paths.extend(natsorted(glob(os.path.join(path, "Validation_Dataset_v1", "Images", "*.tif"))))
        label_paths.extend(natsorted(glob(os.path.join(path, "Validation_Dataset_v1", dirname, "*.tif"))))

    assert raw_paths and len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def get_aimseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    targets: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AimSeg dataset for axon and myelin segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded.
        patch_shape: The patch shape to use for training.
        targets: The choice of support labels for the task.
            Either `instances` for annotated myelinated axons or `semantic` for axons, inner tongue and myelins.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_aimseg_paths(path, None, targets, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs
    )


def get_aimseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    targets: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the AimSeg dataset for axon and myelin segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        targets: The choice of support labels for the task.
            Either 'instances' for annotated myelinated axons or 'semantic' for axons, inner tongue and myelins.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_aimseg_dataset(path, patch_shape, targets, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
