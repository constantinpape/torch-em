"""This dataset contains annotation for 3d fluorescence microscopy segmentation
that were introduced by the EmbedSeg publication.

This dataset is from the publication https://proceedings.mlr.press/v143/lalit21a.html.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util

URLS = {
    "Mouse-Organoid-Cells-CBG": "https://github.com/juglab/EmbedSeg/releases/download/v0.1.0/Mouse-Organoid-Cells-CBG.zip",  # noqa
    "Mouse-Skull-Nuclei-CBG": "https://github.com/juglab/EmbedSeg/releases/download/v0.1.0/Mouse-Skull-Nuclei-CBG.zip",
    "Platynereis-ISH-Nuclei-CBG": "https://github.com/juglab/EmbedSeg/releases/download/v0.1.0/Platynereis-ISH-Nuclei-CBG.zip",  # noqa
    "Platynereis-Nuclei-CBG": "https://github.com/juglab/EmbedSeg/releases/download/v0.1.0/Platynereis-Nuclei-CBG.zip",
}
CHECKSUMS = {
    "Mouse-Organoid-Cells-CBG": "3695ac340473900ace8c37fd7f3ae0d37217de9f2b86c2341f36b1727825e48b",
    "Mouse-Skull-Nuclei-CBG": "3600ec261a48bf953820e0536cacd0bb8a5141be6e7435a4cb0fffeb0caf594e",
    "Platynereis-ISH-Nuclei-CBG": "bc9284df6f6d691a8e81b47310d95617252cc98ebf7daeab55801b330ba921e0",
    "Platynereis-Nuclei-CBG": "448cb7b46f2fe7d472795e05c8d7dfb40f259d94595ad2cfd256bc2aa4ab3be7",
}


def get_embedseg_data(path: Union[os.PathLike, str], name: str, download: bool) -> str:
    """Download the EmbedSeg training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: Name of the dataset to download.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the training data.
    """
    if name not in URLS:
        raise ValueError(f"The dataset name must be in {list(URLS.keys())}. You provided {name}.")

    url = URLS[name]
    checksum = CHECKSUMS[name]

    data_path = os.path.join(path, name)
    if os.path.exists(data_path):
        return data_path

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"{name}.zip")
    util.download_source(zip_path, url, download, checksum)
    util.unzip(zip_path, path, True)

    return data_path


def get_embedseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    name: str,
    split: str = "train",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get an EmbedSeg dataset for 3d fluorescence microscopy segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        name: Name of the dataset to download.
        split: The split to use for the dataset.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    data_root = get_embedseg_data(path, name, download)

    raw_paths = sorted(glob(os.path.join(data_root, split, "images", "*.tif")))
    label_paths = sorted(glob(os.path.join(data_root, split, "masks", "*.tif")))
    assert len(raw_paths) > 0
    assert len(raw_paths) == len(label_paths)

    raw_key, label_key = None, None

    return torch_em.default_segmentation_dataset(
        raw_paths, raw_key, label_paths, label_key, patch_shape, **kwargs
    )


def get_embedseg_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    name: str,
    split: str = "train",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get an EmbedSeg dataloader for 3d fluorescence microscopy segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        name: Name of the dataset to download.
        split: The split to use for the dataset.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_embedseg_dataset(
        path, name=name, split=split, patch_shape=patch_shape, download=download, **ds_kwargs,
    )
    loader = torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
    return loader
