"""The Pancreas FIB-SEM dataset contains training data for segmenting Golgi apparatus and secretory granules
in FIB-SEM volumes of mouse pancreatic beta cells.

It contains two sub-datasets with 3D annotated volumes (isotropic 16 nm resolution):
- golgi: Binary Golgi apparatus masks (8 train volumes of 128^3 voxels, 1 validation volume of 58 x 128 x 128 voxels).
- granules: Instance segmentation of secretory granules (4 train and 1 validation volume of 172^3 voxels).

The dataset is located at https://doi.org/10.5281/zenodo.8114392 and is distributed under the CC BY 4.0 license.
This dataset is from the publication "A modular protocol for segmentation, spatial analysis, and visualization
of volume electron microscopy datasets" (Müller et al.). Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "golgi": "https://zenodo.org/records/8114392/files/data_golgi.zip?download=1",
    "granules": "https://zenodo.org/records/8114392/files/data_granules.zip?download=1",
}
CHECKSUMS = {
    "golgi": "e3c377ba55384351c58b365f71af624d628aeb0a8189ef4fd5ed9a4b2da367a3",
    "granules": "890c903c4df7a16e9f2a84d2ae1fb6b9cd1d81ee39d497d560737933cb5c59bc",
}


def get_pancreas_fibsem_data(
    path: Union[os.PathLike, str], name: Literal["golgi", "granules"], download: bool = False
) -> str:
    """Download the Pancreas FIB-SEM dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the sub-dataset. Either 'golgi' or 'granules'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the data.
    """
    assert name in URLS, f"'{name}' is not a valid sub-dataset name. Choose from {list(URLS)}."

    data_dir = os.path.join(path, f"data_{name}")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"data_{name}.zip")
    util.download_source(path=zip_path, url=URLS[name], download=download, checksum=CHECKSUMS[name])
    util.unzip(zip_path=zip_path, dst=path)
    return data_dir


def get_pancreas_fibsem_paths(
    path: Union[os.PathLike, str],
    name: Literal["golgi", "granules"],
    split: Literal["train", "val"],
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Pancreas FIB-SEM data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the sub-dataset. Either 'golgi' or 'granules'.
        split: The data split to use. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    assert split in ("train", "val"), f"'{split}' is not a valid data split."

    data_dir = get_pancreas_fibsem_data(path, name, download)
    image_paths = natsorted(glob(os.path.join(data_dir, split, "images", "*.tif")))
    label_paths = natsorted(glob(os.path.join(data_dir, split, "masks", "*.tif")))
    assert len(image_paths) > 0 and len(image_paths) == len(label_paths)
    return image_paths, label_paths


def get_pancreas_fibsem_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    name: Literal["golgi", "granules"],
    split: Literal["train", "val"],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Pancreas FIB-SEM dataset for Golgi apparatus or secretory granule segmentation.

    The 'golgi' labels are binary (1 is Golgi apparatus) and the 'granules' labels are instances.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        name: The name of the sub-dataset. Either 'golgi' or 'granules'.
        split: The data split to use. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_pancreas_fibsem_paths(path, name, split, download)

    if name == "granules":
        kwargs, _ = util.add_instance_label_transform(kwargs, add_binary_target=True)
    else:
        kwargs = util.update_kwargs(kwargs, "label_transform", torch_em.transform.label.labels_to_binary)
    kwargs = util.update_kwargs(kwargs, "ndim", 3)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs
    )


def get_pancreas_fibsem_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    name: Literal["golgi", "granules"],
    split: Literal["train", "val"],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for Golgi apparatus or secretory granule segmentation in the Pancreas FIB-SEM dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        name: The name of the sub-dataset. Either 'golgi' or 'granules'.
        split: The data split to use. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pancreas_fibsem_dataset(path, patch_shape, name, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
