"""The Skeletal Muscle FIB-SEM dataset contains FIB-SEM volumes of human skeletal muscle biopsies
with semi-automated segmentations of mitochondria.

The data was acquired from biopsies of 10 individuals (aged 22 to 81) and the standardized volumes were
segmented in a semi-automated manner. In total, 32 standardized volumes of 9 individuals have mitochondria
segmentations.
The dataset is located at https://zenodo.org/records/5873883 (part 1 of 6; the other parts contain the raw data)
and is distributed under the CC BY 4.0 license. It is part of the publication
https://doi.org/10.1016/j.jsb.2019.03.008. Please cite it if you use this dataset in your research.
"""

import os
import re
import warnings
from glob import glob
from shutil import rmtree
from typing import Union, Tuple, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/5873883/files/Human%20Skeletal%20Muscle%201%20of%206.zip/content"
# The md5 checksum of the zip file is 9c43bfbe8971c279b080d0ab59e6afe1, but only sha256 checksums are supported.
CHECKSUM = None

# The volumes of these individuals are used for validation, all other volumes for training.
VAL_DONORS = ("045", "110")


def _volume_key(fname):
    # The naming of the files is not consistent, so we normalize it to match the raw volumes and segmentations.
    stem = os.path.splitext(os.path.basename(fname))[0]
    stem = re.sub(r"[\s_]*\(?mito\)?$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def _find_volume_pairs(names):
    raws, segs = {}, {}
    for name in names:
        parts = name.split("/")
        if parts[0] == "__MACOSX" or not name.lower().endswith(".tif"):
            continue
        donor = re.search(r"GT_?(\d+)", parts[1])
        if donor is None:
            continue
        key = (donor.group(1), _volume_key(name))
        if "Standardised Files" in parts:
            raws[key] = name
        elif "Segmentations" in parts and "z-disk" not in parts[-1].lower():
            segs[key] = name
    return {key: (raws[key], segs[key]) for key in sorted(raws) if key in segs}


def _convert_to_h5(zf, pairs, path):
    import h5py
    import tifffile

    tmp_dir = os.path.join(path, "tmp")
    for (donor, key), (raw_name, seg_name) in pairs.items():
        h5_path = os.path.join(path, f"{key}.h5")
        if os.path.exists(h5_path):
            continue

        raw = tifffile.imread(zf.extract(raw_name, tmp_dir))
        labels = tifffile.imread(zf.extract(seg_name, tmp_dir))
        if raw.shape != labels.shape:
            warnings.warn(f"Skipping {key}: shape of the raw data {raw.shape} and labels {labels.shape} do not match.")
            continue

        chunks = tuple(min(c, s) for c, s in zip((32, 256, 256), raw.shape))
        # NOTE: The segmentations are binary masks, with foreground values 244 and 255.
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip", chunks=chunks)
            f.create_dataset("labels/mitochondria", data=(labels > 0).astype("uint8"), compression="gzip",
                             chunks=chunks)

        rmtree(tmp_dir)


def get_skeletal_muscle_fibsem_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Skeletal Muscle FIB-SEM dataset.

    Note: The data is only available as a 35.9 GB zip archive, which also contains binned volumes, volumes without
    segmentations and Z-disk segmentations that are not used. Only the standardized volumes with mitochondria
    segmentations are extracted and converted to hdf5 files (one per volume), the zip archive is removed afterwards.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the hdf5 files.
    """
    import zipfile

    if len(glob(os.path.join(path, "*.h5"))) > 0:
        return str(path)

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "skeletal_muscle_fibsem.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)

    with zipfile.ZipFile(zip_path) as zf:
        pairs = _find_volume_pairs(zf.namelist())
        _convert_to_h5(zf, pairs, path)

    os.remove(zip_path)
    return str(path)


def get_skeletal_muscle_fibsem_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val"] = "train", download: bool = False
) -> List[str]:
    """Get the paths to the Skeletal Muscle FIB-SEM data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.

    Returns:
        List of the filepaths to the hdf5 files.
    """
    assert split in ("train", "val"), f"'{split}' is not a valid data split."
    get_skeletal_muscle_fibsem_data(path, download)

    h5_paths = sorted(glob(os.path.join(path, "*.h5")))
    is_val = [re.match(r"gt(\d+)", os.path.basename(p)).group(1) in VAL_DONORS for p in h5_paths]
    return [p for p, val in zip(h5_paths, is_val) if val == (split == "val")]


def get_skeletal_muscle_fibsem_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    split: Literal["train", "val"] = "train",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Skeletal Muscle FIB-SEM dataset for mitochondria segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'val'. The split is done by individual.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    paths = get_skeletal_muscle_fibsem_paths(path, split, download)

    kwargs = util.update_kwargs(kwargs, "ndim", 3)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels/mitochondria",
        patch_shape=patch_shape,
        **kwargs
    )


def get_skeletal_muscle_fibsem_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    split: Literal["train", "val"] = "train",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for mitochondria segmentation in the Skeletal Muscle FIB-SEM dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The data split to use. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_skeletal_muscle_fibsem_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
