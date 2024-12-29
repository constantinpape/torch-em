"""The YeaZ dataset contains annotations for yeast cells in brightfield (2d)
and phase-contrast (2d+t) microscopy images.

NOTE: The data is located at:
- Phase-contrast: https://drive.google.com/file/d/14MUIN26ou0L12UC9UV_AC2S3isj1qBMY.
- Brightfield: https://drive.google.com/file/d/1Sot3bau0F0dsBjRxoQzdGOeUy_wMezal

The dataset is located at https://www.epfl.ch/labs/lpbs/data-and-software/.
This dataset is from the publication https://doi.org/10.1038/s41467-020-19557-4.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import json
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = {
    "phc": "https://drive.google.com/file/d/14MUIN26ou0L12UC9UV_AC2S3isj1qBMY",
    "bf": "https://drive.google.com/file/d/1Sot3bau0F0dsBjRxoQzdGOeUy_wMezal"
}


def get_yeaz_data(path: Union[os.PathLike, str], choice: Literal['bf, phc'], download: bool = False) -> str:
    """Obtain the YeaZ dataset.

    NOTE: Please download the dataset manually.

    Args:
        path: Filepath to a folder where the data is expected to be downloaded for further processing.
        download: Whether to download the data if it is not present. Not implemented for this data.

    Returns:
        Filepath where the data is expected to be downloaded.
    """
    if choice not in ['bf', 'phc']:
        raise ValueError(f"'{choice}' is not a valid choice of dataset.")

    data_dir = os.path.join(path, "gold-standard-PhC-plus-2" if choice == "phc" else "gold-standard-BF-V-1")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    tar_path = os.path.join(
        path, "gold-standard-PhC-plus-2.tar.gz" if choice == "phc" else "gold-standard-BF-V-1.tar.gz"
    )

    if not os.path.exists(tar_path) or download:
        raise NotImplementedError(
            f"Automatic download is not supported. Please download the data manually from '{URL[choice]}'."
        )

    util.unzip_tarfile(tar_path=tar_path, dst=path, remove=False)

    return data_dir


def _create_data_splits(path, data_dir, choice, split, raw_paths):
    json_file = os.path.join(path, f"yeaz_{choice}_splits.json")
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
    else:
        # Get the filenames
        names = [os.path.basename(p) for p in raw_paths]

        # Create train / val / test splits
        train_split, test_split = train_test_split(names, test_size=0.2)
        train_split, val_split = train_test_split(train_split, test_size=0.15)
        data = {"train": train_split, "val": val_split, "test": test_split}

        # Write the filenames with splits to a json file.
        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)

    _raw_paths = [os.path.join(data_dir, name) for name in data[split]]
    _label_paths = [p.replace("_im.tif", "_mask.tif") for p in _raw_paths]

    return _raw_paths, _label_paths


def get_yeaz_paths(
    path: Union[os.PathLike, str],
    choice: Literal['bf, phc'],
    split: Literal['train', 'val', 'test'],
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get the YeaZ data.

    Args:
        path: Filepath to a folder where the data is expected to be downloaded for further processing.
        choice: The choice of modality for dataset.
        split: The choice of data split.
        download: Whether to download the data if it is not present. Not implemented for this data.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_yeaz_data(path, choice, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "*_im.tif")))

    # Get the raw and label paths.
    raw_paths, label_paths = _create_data_splits(path, data_dir, choice, split, raw_paths)

    print(len(raw_paths))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_yeaz_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    choice: Literal['bf, phc'],
    split: Literal['train', 'val', 'test'],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the YeaZ dataset for yeast cell segmentation.

    Args:
        path: Filepath to a folder where the data is expected to be downloaded for further processing.
        patch_shape: The patch shape to use for training.
        choice: The choice of modality for dataset.
        split: The choice of data split.
        download: Whether to download the data if it is not present. Not implemented for this data.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_yeaz_paths(path, choice, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs
    )


def get_yeaz_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    choice: Literal['bf, phc'],
    split: Literal['train', 'val', 'test'],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the YeaZ dataloader for yeast cell segmentation.

    Args:
        path: Filepath to a folder where the data is expected to be downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        choice: The choice of modality for dataset.
        split: The choice of data split.
        download: Whether to download the data if it is not present. Not implemented for this data.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_yeaz_dataset(path, patch_shape, choice, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
