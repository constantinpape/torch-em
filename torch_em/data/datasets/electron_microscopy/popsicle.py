"""The POPSICLE dataset contains annotations for bacterial compartment segmentation in cryo-ET.

The data is hosted on the CryoET Data Portal at https://cryoetdataportal.czscience.com/depositions/10350.
It provides curated multi-class compartment masks for 80 tomograms from eight bacterial genera,
together with the matching 20 Angstrom re-binned tomograms, and follows the official train/test split.

The dataset is part of the publication https://doi.org/10.48550/arXiv.2606.10255.
Please cite it if you use this dataset in your research.
"""

import os
import json
import shutil
from typing import Union, Tuple, List, Literal

import requests
from tqdm import tqdm

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://files.cryoetdataportal.cziscience.com/{dataset}/{run}/Reconstructions/VoxelSpacing20.000/"
RAW_URL = BASE_URL + "Tomograms/101/{run}.zarr"
LABEL_URL = BASE_URL + "Annotations/{folder}/{name}-1.0_segmentationmask.zarr"

# The compartment classes with their label value in the merged mask and their portal annotation folder.
CLASSES = {
    "cytoplasm": (1, "100"),
    "membrane": (2, "102"),
    "periplasmic_space": (3, "104"),
    "bacterial_type_flagellum": (4, "101"),
    "dense_body": (5, "103"),
}
CORE_CLASSES = ("cytoplasm", "membrane", "periplasmic_space")
OPTIONAL_CLASSES = {"f": "bacterial_type_flagellum", "d": "dense_body"}

# The portal dataset, run name, official split and the optional classes of each run.
RUNS = [
    (10053, "dga2017-01-14-21", "test", ""),
    (10054, "dga2016-09-09-26", "train", ""),
    (10054, "dga2016-09-09-32", "train", ""),
    (10054, "dga2016-09-09-33", "train", ""),
    (10054, "dga2016-09-09-35", "train", ""),
    (10054, "dga2016-09-09-42", "train", ""),
    (10054, "dga2016-09-09-51", "train", ""),
    (10054, "dga2016-09-09-6", "train", ""),
    (10054, "dga2016-09-09-78", "train", ""),
    (10065, "dga2016-01-13-17", "test", "fd"),
    (10065, "dga2016-01-13-25", "train", "fd"),
    (10065, "dga2016-01-13-26", "train", "fd"),
    (10065, "dga2016-01-13-30", "train", "fd"),
    (10065, "dga2016-01-13-32", "train", "d"),
    (10098, "dga2015-10-29-11", "test", "d"),
    (10098, "dga2015-10-29-31", "train", "d"),
    (10098, "dga2015-10-29-39", "train", "d"),
    (10098, "dga2015-10-29-43", "train", "d"),
    (10098, "dga2015-10-29-60", "train", "d"),
    (10155, "ycw2012-11-14-20", "test", "fd"),
    (10155, "ycw2012-11-14-47", "train", "fd"),
    (10155, "ycw2012-11-14-59", "train", "fd"),
    (10155, "ycw2013-05-01-34", "train", "f"),
    (10155, "ycw2013-08-20-24", "train", "fd"),
    (10155, "ycw2013-08-20-28", "train", "fd"),
    (10155, "ycw2013-08-20-43", "train", "f"),
    (10155, "ycw2013-08-20-45", "train", "fd"),
    (10155, "ycw2013-08-20-59", "train", "fd"),
    (10155, "ycw2013-09-10-28", "test", "fd"),
    (10155, "ycw2013-09-10-36", "train", "d"),
    (10155, "ycw2013-09-10-39", "train", "f"),
    (10155, "ycw2013-09-10-43", "train", "fd"),
    (10155, "ycw2013-09-10-47", "train", "fd"),
    (10155, "ycw2013-09-10-48", "train", "fd"),
    (10161, "ycw2012-09-23-21", "test", "f"),
    (10161, "ycw2012-09-23-31", "train", "f"),
    (10161, "ycw2012-09-23-39", "train", "f"),
    (10161, "ycw2012-09-23-46", "train", "f"),
    (10161, "ycw2012-09-23-55", "train", "f"),
    (10161, "ycw2012-09-23-66", "train", "f"),
    (10161, "ycw2012-09-23-67", "train", "f"),
    (10161, "ycw2012-09-23-70", "train", "f"),
    (10162, "ycw2012-03-12-3", "train", ""),
    (10163, "ycw2012-03-03-2", "test", "d"),
    (10163, "ycw2012-03-03-3", "train", ""),
    (10163, "ycw2012-03-12-18", "train", ""),
    (10163, "ycw2012-03-12-7", "train", ""),
    (10166, "ycw2012-09-07-23", "train", ""),
    (10166, "ycw2012-09-07-3", "train", ""),
    (10166, "ycw2012-10-08-1", "train", ""),
    (10166, "ycw2012-10-08-7", "train", ""),
    (10226, "mba2011-11-23-1", "test", "f"),
    (10226, "mba2011-11-23-15", "train", "f"),
    (10226, "mba2011-11-23-16", "train", "f"),
    (10226, "mba2011-11-23-20", "train", "f"),
    (10226, "mba2011-11-23-21", "train", "f"),
    (10226, "mba2011-11-23-22", "test", "f"),
    (10226, "mba2011-11-23-25", "train", "f"),
    (10226, "mba2011-11-23-26", "train", "f"),
    (10226, "mba2011-11-23-35", "train", "f"),
    (10226, "mba2011-11-23-7", "train", "f"),
    (10272, "aba2015-06-04-10", "test", "f"),
    (10272, "aba2015-06-04-16", "test", ""),
    (10272, "aba2015-06-04-22", "train", "d"),
    (10272, "aba2015-06-04-24", "train", ""),
    (10272, "aba2015-06-04-26", "train", ""),
    (10272, "aba2015-06-04-30", "train", ""),
    (10272, "aba2015-06-04-5", "train", "d"),
    (10272, "aba2015-06-04-9", "train", ""),
    (10273, "aba2015-07-07-10", "train", "d"),
    (10273, "aba2015-07-07-17", "train", "f"),
    (10281, "aba2015-02-23-15", "train", "d"),
    (10281, "aba2015-02-23-16", "train", "d"),
    (10281, "aba2015-02-23-2", "test", "fd"),
    (10281, "aba2015-02-23-20", "train", "fd"),
    (10281, "aba2015-02-23-21", "train", "d"),
    (10281, "aba2015-02-23-25", "train", "d"),
    (10281, "aba2015-02-23-3", "train", "fd"),
    (10281, "aba2015-02-23-30", "train", "d"),
    (10281, "aba2015-02-23-9", "train", "fd"),
]


def _fetch(url, path, optional=False):
    if os.path.exists(path):
        return True

    with requests.get(url, stream=True, timeout=(20, 300)) as response:
        # A chunk that holds only the fill value is not written by the portal.
        if optional and response.status_code == 404:
            return False
        response.raise_for_status()
        # The chunk is renamed only once it is complete, so an interrupted download is not reused.
        tmp_path = path + ".partial"
        with open(tmp_path, "wb") as f:
            for block in response.iter_content(8 * 1024 ** 2):
                f.write(block)

    os.rename(tmp_path, path)
    return True


def _download_ome_zarr(url, out_path, download):
    array_path = os.path.join(out_path, "0")
    if os.path.exists(array_path):
        return array_path

    if not download:
        raise RuntimeError(f"Cannot find the data at {out_path}, but download was set to False.")

    os.makedirs(out_path, exist_ok=True)
    for name in (".zattrs", ".zgroup"):
        if not os.path.exists(os.path.join(out_path, name)):
            _fetch(f"{url}/{name}", os.path.join(out_path, name))

    tmp_path = os.path.join(out_path, "0.partial")
    os.makedirs(tmp_path, exist_ok=True)
    _fetch(f"{url}/0/.zarray", os.path.join(tmp_path, ".zarray"))
    with open(os.path.join(tmp_path, ".zarray")) as f:
        meta = json.load(f)

    grid = [-(-size // chunk) for size, chunk in zip(meta["shape"], meta["chunks"])]
    for z in range(grid[0]):
        for y in range(grid[1]):
            for x in range(grid[2]):
                chunk_dir = os.path.join(tmp_path, str(z), str(y))
                os.makedirs(chunk_dir, exist_ok=True)
                _fetch(f"{url}/0/{z}/{y}/{x}", os.path.join(chunk_dir, str(x)), optional=True)

    os.rename(tmp_path, array_path)
    return array_path


def _merge_labels(run_dir, dataset, run, extras, download):
    """Merge the per-class masks into one multi-class volume.

    The classes are only mutually exclusive at the native resolution, so the coarser levels of the
    portal label pyramid are not used.
    """
    import zarr

    label_path = os.path.join(run_dir, "labels.zarr")
    if os.path.exists(label_path):
        return label_path

    names = list(CORE_CLASSES) + [OPTIONAL_CLASSES[key] for key in extras]
    merged = None
    for name in names:
        value, folder = CLASSES[name]
        url = LABEL_URL.format(dataset=dataset, run=run, folder=folder, name=name)
        class_dir = os.path.join(run_dir, f"class_{name}.zarr")
        array = zarr.open_array(_download_ome_zarr(url, class_dir, download), mode="r")
        mask = array[:] > 0
        if merged is None:
            merged = np.zeros(mask.shape, dtype="uint8")
        elif mask.shape != merged.shape:
            raise ValueError(f"The class masks of run {run} have different shapes.")
        if np.any(merged[mask]):
            raise ValueError(f"The class masks of run {run} overlap, so they cannot be merged.")
        merged[mask] = value

    # The store is built under a temporary name and renamed, so an interrupted merge leaves nothing behind.
    # It uses the zarr format of the portal stores it sits next to.
    tmp_path = label_path + ".partial"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    store = zarr.open_group(tmp_path, mode="w", zarr_format=2)
    array = store.create_array("0", shape=merged.shape, dtype="uint8", chunks=(64, 256, 256))
    array[:] = merged
    os.rename(tmp_path, label_path)
    return label_path


def get_popsicle_data(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> str:
    """Download the POPSICLE bacterial segmentation dataset.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        split: The data split to download. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    if split not in ("train", "test"):
        raise ValueError(f"The split must be 'train' or 'test', got '{split}'.")

    data_dir = os.path.join(path, split)
    os.makedirs(data_dir, exist_ok=True)

    runs = [entry for entry in RUNS if entry[2] == split]
    for dataset, run, _, extras in tqdm(runs, desc=f"Downloading the {split} tomograms"):
        run_dir = os.path.join(data_dir, run)
        _download_ome_zarr(RAW_URL.format(dataset=dataset, run=run), os.path.join(run_dir, "raw.zarr"), download)
        if not download and not os.path.exists(os.path.join(run_dir, "labels.zarr")):
            raise RuntimeError(f"Cannot find the data at {run_dir}, but download was set to False.")
        _merge_labels(run_dir, dataset, run, extras, download)

    return data_dir


def get_popsicle_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the POPSICLE data.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the tomograms.
        List of filepaths to the multi-class compartment masks.
    """
    data_dir = get_popsicle_data(path, split, download)
    runs = [entry[1] for entry in RUNS if entry[2] == split]
    raw_paths = [os.path.join(data_dir, run, "raw.zarr") for run in runs]
    label_paths = [os.path.join(data_dir, run, "labels.zarr") for run in runs]
    return raw_paths, label_paths


def get_popsicle_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    split: Literal["train", "test"],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the dataset for bacterial compartment segmentation in cryo-ET data.

    The labels are a multi-class mask with 1: cytoplasm, 2: membrane, 3: periplasmic space,
    4: bacterial-type flagellum and 5: dense body. The last two classes are not present in every tomogram.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        patch_shape: The patch shape to use for training.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    raw_paths, label_paths = get_popsicle_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="0",
        label_paths=label_paths,
        label_key="0",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_popsicle_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    split: Literal["train", "test"],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for bacterial compartment segmentation in cryo-ET data.

    Args:
        path: Filepath to a folder where the data will be downloaded.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_popsicle_dataset(path, patch_shape, split, download=download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
