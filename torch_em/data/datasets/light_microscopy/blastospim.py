"""The BlastoSPIM dataset contains annotations for nucleus segmentation in
selective plane illumination microscopy (SPIM) images of preimplantation mouse embryo.

This dataset is from the publication https://doi.org/10.1242/dev.202817.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Tuple, List, Union

import gzip
import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://plus.figshare.com/ndownloader/articles/26540593/versions/1"
CHECKSUM = "8be979c5a06cfad479a5cfe21b8bbb0e26f0e677cb052fe43275fa451fa9e9ac"


def _preprocess_inputs(data_dir):
    import h5py

    raw_paths = natsorted(glob(os.path.join(data_dir, "*_image_*.npy.gz")))
    label_paths = natsorted(glob(os.path.join(data_dir, "*_masks_*.npy.gz")))

    preprocessed_dir = os.path.join(data_dir, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    for rpath, lpath in tqdm(
        zip(raw_paths, label_paths), desc="Preprocessing inputs", total=len(raw_paths)
    ):
        with gzip.open(rpath, "rb") as fr:
            raw = np.load(fr)

        with gzip.open(lpath, "rb") as fl:
            labels = np.load(fl)

        vname = os.path.basename(rpath).split(".")[0]
        volume_path = os.path.join(preprocessed_dir, Path(vname).with_suffix(".h5"))
        with h5py.File(volume_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_blastospim_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BlastoSPIM dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded and preprocessed.
    """
    data_dir = os.path.join(path, "data", "preprocessed")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "data.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=os.path.join(path, "data"))

    # Preprocess inputs.
    _preprocess_inputs(os.path.join(path, "data"))

    return data_dir


def get_blastospim_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the BlastoSPIM data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    data_dir = get_blastospim_data(path, download)
    volume_paths = glob(os.path.join(data_dir, "*.h5"))
    return volume_paths


def get_blastospim_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, ...], download: bool = False, **kwargs
) -> Dataset:
    """Download the BlastoSPIM dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_blastospim_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_blastospim_loader(
    path: Union[os.PathLike, str], batch_size: int, patch_shape: Tuple[int, ...], download: bool = False, **kwargs
) -> DataLoader:
    """Download the BlastoSPIM dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_blastospim_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
