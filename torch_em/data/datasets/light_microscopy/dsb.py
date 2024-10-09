"""This Dataset was used in a Kaggle Data Science Bowl. It contains light microscopy
images with annotations for nucleus segmentation.

The dataset is described in the publication https://doi.org/10.1038/s41592-019-0612-7.
Please cite it if you use this dataset in your research.
"""

import os
from shutil import move
from typing import List, Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


DSB_URLS = {
    "full": "",  # TODO
    "reduced": "https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip"
}
CHECKSUMS = {
    "full": None,
    "reduced": "e44921950edce378063aa4457e625581ba35b4c2dbd9a07c19d48900129f386f"
}


def get_dsb_data(path: Union[os.PathLike, str], source: str, download: bool) -> str:
    """Download the DSB training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        source: The source of the dataset. Can either be 'full' for the complete dataset,
            or 'reduced' for the dataset excluding histopathology images.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the training data.
    """
    os.makedirs(path, exist_ok=True)
    url = DSB_URLS[source]
    checksum = CHECKSUMS[source]

    train_out_path = os.path.join(path, "train")
    test_out_path = os.path.join(path, "test")

    if os.path.exists(train_out_path) and os.path.exists(test_out_path):
        return path

    zip_path = os.path.join(path, "dsb.zip")
    util.download_source(zip_path, url, download, checksum)
    util.unzip(zip_path, path, True)

    move(os.path.join(path, "dsb2018", "train"), train_out_path)
    move(os.path.join(path, "dsb2018", "test"), test_out_path)
    return path


def get_dsb_paths(path: Union[os.PathLike, str], split: str, source: str, download: bool = False) -> Tuple[str, str]:
    """Get paths to the DSB data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use for the dataset. Either 'train' or 'test'.
        source: The source of the dataset. Can either be 'full' for the complete dataset,
            or 'reduced' for the dataset excluding histopathology images.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath for the folder where the images are stored.
        Filepath for the folder where the labels are stored.
    """
    get_dsb_data(path, source, download)

    image_path = os.path.join(path, split, "images")
    label_path = os.path.join(path, split, "masks")

    return image_path, label_path


def get_dsb_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    source: str = "reduced",
    **kwargs
) -> Dataset:
    """Get the DSB dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use for the dataset. Either 'train' or 'test'.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        source: The source of the dataset. Can either be 'full' for the complete dataset,
            or 'reduced' for the dataset excluding histopathology images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    assert split in ("test", "train"), split

    image_path, label_path = get_dsb_paths(path, split, source, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )
    kwargs = util.update_kwargs(kwargs, "ndim", 2)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_path,
        raw_key="*.tif",
        label_paths=label_path,
        label_key="*.tif",
        patch_shape=patch_shape,
        **kwargs
    )


def get_dsb_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    source: str = "reduced",
    **kwargs
) -> DataLoader:
    """Get the DSB dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use for the dataset. Either 'train' or 'test'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        source: The source of the dataset. Can either be 'full' for the complete dataset,
            or 'reduced' for the dataset excluding histopathology images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_dsb_dataset(
        path, split, patch_shape, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary,
        source=source, **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
