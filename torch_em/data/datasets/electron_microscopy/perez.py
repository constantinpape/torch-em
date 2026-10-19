"""The Perez dataset contains annotations for semantic segmentation of mitochondria, lysosomes, nuclei and nucleoli
in serial block-face scanning electron microscopy (SBF-SEM) images of the mouse suprachiasmatic nucleus.

Each organelle is a separate binary segmentation task, with its own set of 2D tiles.

The original data is from the publication https://doi.org/10.3389/fnana.2014.00126.
The preprocessed version used here was released with the RETINA benchmark: https://github.com/BaderLab/RETINA
(publication: https://doi.org/10.1371/journal.pcbi.1013115).
Please cite both publications if you use this dataset in your research.
"""

import os
from glob import glob
from zipfile import ZipFile
from typing import List, Literal, Tuple, Union

from natsort import natsorted
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


# NOTE: This is the full RETINA benchmark archive (~1.2 GB), which contains four other datasets besides Perez.
# Only the Perez files are extracted.
URL = "https://drive.google.com/uc?id=1T0fShblGYgKxTRwPlbMSvMbHQqmEk8YJ"
CHECKSUM = "e65555a039af5cacd64d72209f34a31c2af7afcdb887b5420e3ef38401ba3f45"

ORGANELLES = ["mito", "lyso", "nuclei", "nucleoli"]


def get_perez_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Perez dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data.
    """
    data_dir = os.path.join(path, "perez")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "benchmark_dataset.zip")
    util.download_source_gdrive(zip_path, URL, download, checksum=CHECKSUM)

    prefix = "benchdata/perez/"
    with ZipFile(zip_path) as f:
        members = [m for m in f.namelist() if m.startswith(prefix) and not m.endswith(".DS_Store")]
        f.extractall(path, members=members)

    os.rename(os.path.join(path, "benchdata", "perez"), data_dir)
    os.rmdir(os.path.join(path, "benchdata"))
    os.remove(zip_path)

    return data_dir


def get_perez_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    organelle: Literal["mito", "lyso", "nuclei", "nucleoli"] = "mito",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Perez data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train' or 'test'.
        organelle: The organelle to segment. Either 'mito', 'lyso', 'nuclei' or 'nucleoli'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in ["train", "test"]:
        raise ValueError(f"'{split}' is not a valid split. Please choose from 'train' / 'test'.")

    if organelle not in ORGANELLES:
        raise ValueError(f"'{organelle}' is not a valid organelle. Please choose from {ORGANELLES}.")

    data_dir = get_perez_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, organelle, split, "images", "*.png")))
    label_paths = natsorted(glob(os.path.join(data_dir, organelle, split, "masks", "*.png")))
    assert raw_paths and len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def get_perez_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    organelle: Literal["mito", "lyso", "nuclei", "nucleoli"] = "mito",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Perez dataset for organelle segmentation in SBF-SEM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split. Either 'train' or 'test'.
        organelle: The organelle to segment. Either 'mito', 'lyso', 'nuclei' or 'nucleoli'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_perez_paths(path, split, organelle, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_perez_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    organelle: Literal["mito", "lyso", "nuclei", "nucleoli"] = "mito",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Perez dataloader for organelle segmentation in SBF-SEM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split. Either 'train' or 'test'.
        organelle: The organelle to segment. Either 'mito', 'lyso', 'nuclei' or 'nucleoli'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_perez_dataset(path, patch_shape, split, organelle, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
