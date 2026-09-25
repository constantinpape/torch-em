"""The SurfaceMorphometrics dataset contains voxel segmentations of mitochondria and endoplasmic
reticulum membranes in cryo-electron tomograms of mouse embryonic fibroblasts, treated with vehicle
or thapsigargin.

Each tomogram has a multi-class semantic label volume (0=background, 1-3=membrane classes). The
source deposit does not document a fixed name for each label value; inspect a representative volume
to determine which value corresponds to which membrane in your use case.

The data is available at https://www.ebi.ac.uk/empiar/EMPIAR-11370/.
The dataset was published in https://doi.org/10.1101/2022.01.23.477440.
Please cite this publication if you use the dataset in your research.
"""

import os
from typing import List, Tuple, Union

import numpy as np

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


BASE_URL = "https://ftp.ebi.ac.uk/empiar/world_availability/11370/data"

TOMOGRAM_IDS = (
    "TE2", "TE3", "TE4", "TE5", "TE6", "TE7", "TE8", "TE9", "TE10", "TE11", "TE12", "TE13", "TE14",
    "TF1", "TF2", "TF3", "TF5", "TF6",
    "UE1", "UE2", "UE3", "UE4", "UE5", "UE6", "UE7", "UE8", "UE9", "UE10",
    "UF1", "UF2", "UF3", "UF4", "UF5", "UF6",
)

# UE9 and UF3 were deposited without the usual '.mrc' suffix on their label file.
_LABEL_SUFFIX_EXCEPTIONS = {"UE9": "_labels.rec", "UF3": "_labels.rec"}


def _label_url(tomogram_id):
    suffix = _LABEL_SUFFIX_EXCEPTIONS.get(tomogram_id, "_labels.rec.mrc")
    return f"{BASE_URL}/Voxel_Segmentations/{tomogram_id}{suffix}"


def _read_raw_mrc(path):
    """Read an MRC volume, tolerating the non-standard headers in this deposit."""
    import mrcfile
    with mrcfile.open(path, permissive=True) as mrc:
        return np.asarray(mrc.data)


def _read_label_mrc(path, shape):
    """Read a label MRC volume as raw uint8 bytes after the standard 1024-byte header.

    `mrcfile` rejects these files outright (non-standard machine stamp, missing map ID), so the
    label data is read directly instead.
    """
    with open(path, "rb") as f:
        content = f.read()
    voxel_count = int(np.prod(shape))
    return np.frombuffer(content[1024:1024 + voxel_count], dtype=np.uint8).reshape(shape)


def get_surface_morphometrics_data(
    path: Union[os.PathLike, str], tomogram_id: str, download: bool = False,
) -> Tuple[str, str]:
    """Download one tomogram and its label volume from the SurfaceMorphometrics dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        tomogram_id: Which tomogram to download. One of `TOMOGRAM_IDS`.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the downloaded tomogram.
        The filepath to the downloaded label volume.
    """
    if tomogram_id not in TOMOGRAM_IDS:
        raise ValueError(f"tomogram_id must be one of {TOMOGRAM_IDS}, got {tomogram_id!r}")

    os.makedirs(path, exist_ok=True)
    raw_path = os.path.join(path, f"{tomogram_id}_tomo.rec.mrc")
    label_path = os.path.join(path, f"{tomogram_id}_labels.rec")

    util.download_source(raw_path, f"{BASE_URL}/Tomograms/{tomogram_id}_tomo.rec.mrc", download, checksum=None)
    util.download_source(label_path, _label_url(tomogram_id), download, checksum=None)

    return raw_path, label_path


def get_surface_morphometrics_paths(
    path: Union[os.PathLike, str],
    tomogram_ids: List[str],
    cache_path: Union[os.PathLike, str, None] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to cached SurfaceMorphometrics zarr stores, one per tomogram.

    Args:
        path: Filepath to a folder where the downloaded raw MRC files will be saved.
        tomogram_ids: Which tomograms to use. See `TOMOGRAM_IDS`.
        cache_path: Filepath to a folder where the converted zarr stores will be saved. Defaults to
            `path` if not given.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    import zarr
    from zarr.codecs import BloscCodec

    cache_path = cache_path or path
    os.makedirs(cache_path, exist_ok=True)

    zarr_paths = []
    for tomogram_id in tomogram_ids:
        zarr_path = os.path.join(str(cache_path), f"{tomogram_id}.zarr")
        root = zarr.open_group(zarr_path, mode="a")
        if "raw" not in root or "labels" not in root:
            raw_path, label_path = get_surface_morphometrics_data(path, tomogram_id, download)
            raw = _read_raw_mrc(raw_path)
            labels = _read_label_mrc(label_path, raw.shape)

            def _make_array(name, data, shuffle):
                array = root.create_array(
                    name, shape=data.shape, chunks=(32, 256, 256), dtype=data.dtype,
                    compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
                )
                array[:] = data

            _make_array("raw", raw, shuffle="shuffle")
            _make_array("labels", labels, shuffle="bitshuffle")
        zarr_paths.append(zarr_path)

    return zarr_paths


def get_surface_morphometrics_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    tomogram_ids: List[str] = TOMOGRAM_IDS,
    cache_path: Union[os.PathLike, str, None] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the SurfaceMorphometrics dataset for mitochondria and ER membrane segmentation.

    Args:
        path: Filepath to a folder where the downloaded raw MRC files will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        tomogram_ids: Which tomograms to use. See `TOMOGRAM_IDS`. Defaults to all of them.
        cache_path: Filepath to a folder where the converted zarr stores will be saved. Defaults to
            `path` if not given.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_surface_morphometrics_paths(path, tomogram_ids, cache_path, download)
    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_surface_morphometrics_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    tomogram_ids: List[str] = TOMOGRAM_IDS,
    cache_path: Union[os.PathLike, str, None] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for mitochondria and ER membrane segmentation in the SurfaceMorphometrics
    dataset.

    Args:
        path: Filepath to a folder where the downloaded raw MRC files will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        tomogram_ids: Which tomograms to use. See `TOMOGRAM_IDS`. Defaults to all of them.
        cache_path: Filepath to a folder where the converted zarr stores will be saved. Defaults to
            `path` if not given.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the
            PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_surface_morphometrics_dataset(
        path, patch_shape, tomogram_ids, cache_path, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
