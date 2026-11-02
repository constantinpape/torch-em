"""The BEACON dataset contains nucleus annotations for H&E stained biopsies of Barrett's esophagus.

The cell segmentation part of BEACON holds one tile per slide, 67 tiles in total, from two cohorts:
MD Anderson ('MDA', the discovery cohort) and Northwestern University ('NU', the test cohort).
One pathologist outlined every nucleus by hand in QuPath and assigned it to 'epithelial' or 'other'.
NOTE: The tiles on the server are 257 x 257 pixels, although the dataset README states 512 x 512 pixels.
NOTE: A few nuclei per tile carry a third, generic class 'Cell'. This module maps them to 'other'.

The dataset is located at https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD4181 (CC BY 4.0).
This dataset is from the publication https://doi.org/10.1101/2025.11.11.25339952.
Please cite it if you use this dataset for your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

import numpy as np
import pandas as pd
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/181/S-BIAD4181/Files"
FILE_LIST = "file_lists/annotation/cell_segmentation.tsv"
COHORTS = {"MDA (Discovery)": "mda", "NU (Test)": "nu"}
CLASSES = {"epithelial": 1, "other": 2, "Cell": 2}


def _download_tiles(path, download):
    tsv_path = os.path.join(path, "cell_segmentation.tsv")
    util.download_source(tsv_path, f"{BASE_URL}/{FILE_LIST}", download, checksum=None)
    table = pd.read_csv(tsv_path, sep="\t")

    tiles = {}
    for files, source, cohort in zip(table["Files"], table["source_image"], table["Cohort"]):
        for remote in (files, source):
            local = os.path.join(path, "raw", os.path.basename(remote))
            if not os.path.exists(local):
                os.makedirs(os.path.dirname(local), exist_ok=True)
                util.download_source(local, f"{BASE_URL}/{remote}", download, checksum=None)
        tiles[os.path.splitext(os.path.basename(source))[0]] = COHORTS[cohort]
    return tiles


def _create_h5_files(path, tiles):
    import h5py

    for tile, cohort in tqdm(tiles.items(), desc="Preprocess BEACON tiles"):
        out_dir = os.path.join(path, "preprocessed", cohort)
        os.makedirs(out_dir, exist_ok=True)

        image = imageio.imread(os.path.join(path, "raw", f"{tile}.png"))[..., :3]
        instances = imageio.imread(os.path.join(path, "raw", f"{tile}_mask.tif")).astype("uint32")
        with open(os.path.join(path, "raw", f"{tile}_classmap.json")) as f:
            classmap = json.load(f)

        semantic = np.zeros_like(instances, dtype="uint8")
        for name, ids in classmap.items():
            semantic[np.isin(instances, ids)] = CLASSES[name]

        with h5py.File(os.path.join(out_dir, f"{tile}.h5"), "w") as f:
            f.create_dataset("raw", data=image.transpose(2, 0, 1), compression="gzip")
            f.create_dataset("labels/instances", data=instances, compression="gzip")
            f.create_dataset("labels/semantic", data=semantic, compression="gzip")


def get_beacon_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BEACON nucleus segmentation data.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    data_dir = os.path.join(path, "preprocessed")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    tiles = _download_tiles(path, download)
    _create_h5_files(path, tiles)
    return data_dir


def get_beacon_paths(
    path: Union[os.PathLike, str],
    cohort: Optional[Literal["mda", "nu"]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the BEACON data.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        cohort: The choice of cohort, 'mda' or 'nu'. By default, the tiles of both cohorts are returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    data_dir = get_beacon_data(path, download)
    if cohort is not None:
        assert cohort in ("mda", "nu"), cohort
    return natsorted(glob(os.path.join(data_dir, cohort or "*", "*.h5")))


def get_beacon_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    cohort: Optional[Literal["mda", "nu"]] = None,
    label_choice: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BEACON dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        patch_shape: The patch shape to use for training.
        cohort: The choice of cohort, 'mda' or 'nu'. By default, the tiles of both cohorts are used.
        label_choice: The choice of labels, 'instances' or 'semantic' (1: epithelial, 2: other).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert label_choice in ("instances", "semantic"), label_choice
    paths = get_beacon_paths(path, cohort, download)
    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key=f"labels/{label_choice}",
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs
    )


def get_beacon_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    cohort: Optional[Literal["mda", "nu"]] = None,
    label_choice: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BEACON dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        cohort: The choice of cohort, 'mda' or 'nu'. By default, the tiles of both cohorts are used.
        label_choice: The choice of labels, 'instances' or 'semantic' (1: epithelial, 2: other).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_beacon_dataset(path, patch_shape, cohort, label_choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
