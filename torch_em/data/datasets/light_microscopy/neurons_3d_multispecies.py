"""This dataset contains 3D label-free microscopy volumes of neuronal cell
bodies (somata) from human, mouse and rat brain tissue, imaged with oblique
illumination or Dodt gradient contrast (DGC) microscopy, with manually
annotated instance segmentation masks and an official train / val / test split.

The dataset is located at https://zenodo.org/records/20797635.
Please cite it if you use this dataset in your research.
"""

import os
import json
from glob import glob
from natsort import natsorted
from typing import List, Literal, Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "human_oblique": "https://zenodo.org/records/20797635/files/human_neurons_oblique.zip",
    "human_dodt": "https://zenodo.org/records/20797635/files/human_neurons_dodt.zip",
    "mouse_oblique": "https://zenodo.org/records/20797635/files/mice_neurons_oblique.zip",
    "mouse_dodt": "https://zenodo.org/records/20797635/files/mice_neurons_dodt.zip",
    "rat_oblique": "https://zenodo.org/records/20797635/files/rat_neurons_oblique.zip",
    "rat_dodt": "https://zenodo.org/records/20797635/files/rat_neurons_dodt.zip",
}

CHECKSUMS = {
    "human_oblique": "8b2c37b74fe2b890a3d941097d619651d56fa457a37c010c1c31d24604acdebc",
    "human_dodt": "1b4a72302e0e85dfb3169fd753dcfca6c4d200cea135ce7b2d8401af009d3621",
    "mouse_oblique": "31dd601cf114359a66e39266b688e51448f70bf6a516474b8905f3071ce372fa",
    "mouse_dodt": "3debfdd5534509b8161bbf2f952cba1bd7629aa22c78d5f32bf585d10edb3b1c",
    "rat_oblique": "0b7be7b0a9436d74ed1f2f2676256c9ad0072996f89fd95ce2b01a4a2cc64f86",
    "rat_dodt": "431269c513bb558f7434018fc087c22e31c1e7cbdccc4da2d399d0bf3b85715a",
}

# Zenodo archive names differ from the species / modality keys used here.
ARCHIVE_NAMES = {
    "human_oblique": "human_neurons_oblique",
    "human_dodt": "human_neurons_dodt",
    "mouse_oblique": "mice_neurons_oblique",
    "mouse_dodt": "mice_neurons_dodt",
    "rat_oblique": "rat_neurons_oblique",
    "rat_dodt": "rat_neurons_dodt",
}

SPECIES = ["human", "mouse", "rat"]
MODALITIES = ["oblique", "dodt"]


def get_neurons_3d_multispecies_data(
    path: Union[os.PathLike, str],
    species: Optional[List[Literal["human", "mouse", "rat"]]] = None,
    modality: Optional[List[Literal["oblique", "dodt"]]] = None,
    download: bool = False,
) -> List[str]:
    """Download the multi-species 3D neuron segmentation dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        species: The species subset(s) to download. Defaults to all species.
        modality: The imaging modality subset(s) to download. Defaults to all modalities.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the extracted data directories.
    """
    species = SPECIES if species is None else species
    modality = MODALITIES if modality is None else modality

    os.makedirs(path, exist_ok=True)

    data_dirs = []
    for sp in species:
        for mod in modality:
            key = f"{sp}_{mod}"
            data_dir = os.path.join(path, ARCHIVE_NAMES[key])
            if not os.path.exists(data_dir):
                zip_path = os.path.join(path, f"{ARCHIVE_NAMES[key]}.zip")
                util.download_source(zip_path, URLS[key], download, checksum=CHECKSUMS[key])
                util.unzip(zip_path, path)
            data_dirs.append(data_dir)

    return data_dirs


def get_neurons_3d_multispecies_paths(
    path: Union[os.PathLike, str],
    species: Optional[List[Literal["human", "mouse", "rat"]]] = None,
    modality: Optional[List[Literal["oblique", "dodt"]]] = None,
    split: Optional[Literal["train", "val", "test"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the multi-species 3D neuron segmentation data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        species: The species subset(s) to use. Defaults to all species.
        modality: The imaging modality subset(s) to use. Defaults to all modalities.
        split: The data split to use. Either 'train', 'val' or 'test'. Defaults to using all splits.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dirs = get_neurons_3d_multispecies_data(path, species, modality, download)

    raw_paths, label_paths = [], []
    for data_dir in data_dirs:
        if split is None:
            fnames = natsorted(os.path.basename(p) for p in glob(os.path.join(data_dir, "images", "*.tif")))
        else:
            split_file = os.path.join(data_dir, "split.json")
            with open(split_file) as f:
                fnames = natsorted(json.load(f)[split])

        for fname in fnames:
            raw_path = os.path.join(data_dir, "images", fname)
            label_path = os.path.join(data_dir, "masks", fname)
            if os.path.exists(raw_path) and os.path.exists(label_path):
                raw_paths.append(raw_path)
                label_paths.append(label_path)

    if len(raw_paths) == 0:
        raise RuntimeError(f"No image files found under {path}. Please check the dataset structure.")

    return raw_paths, label_paths


def get_neurons_3d_multispecies_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    species: Optional[List[Literal["human", "mouse", "rat"]]] = None,
    modality: Optional[List[Literal["oblique", "dodt"]]] = None,
    split: Optional[Literal["train", "val", "test"]] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the multi-species 3D neuron segmentation dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        species: The species subset(s) to use. Defaults to all species.
        modality: The imaging modality subset(s) to use. Defaults to all modalities.
        split: The data split to use. Either 'train', 'val' or 'test'. Defaults to using all splits.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_neurons_3d_multispecies_paths(path, species, modality, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_neurons_3d_multispecies_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    species: Optional[List[Literal["human", "mouse", "rat"]]] = None,
    modality: Optional[List[Literal["oblique", "dodt"]]] = None,
    split: Optional[Literal["train", "val", "test"]] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the multi-species 3D neuron segmentation dataloader.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        species: The species subset(s) to use. Defaults to all species.
        modality: The imaging modality subset(s) to use. Defaults to all modalities.
        split: The data split to use. Either 'train', 'val' or 'test'. Defaults to using all splits.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_neurons_3d_multispecies_dataset(path, patch_shape, species, modality, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
