"""The SXT-INS1E-Mito dataset contains organelle segmentation for whole INS-1E pancreatic beta cells
imaged by cryo-hydrated soft X-ray tomography (SXT), across unstimulated and glucose/Exendin-4
stimulated conditions.

IMPORTANT: The imaging modality is soft X-ray tomography, not electron microscopy. It is filed here
because it shares the same volumetric organelle-segmentation role as the other datasets in this module,
not because it is EM.

Each of the 55 cells has a semantic label volume with four classes:
    0: exterior, 1: cell, 2: nucleus, 5: mitochondria

The data is available at https://doi.org/10.5281/zenodo.20513085 under the CC-BY-4.0 license.
The dataset was published in https://doi.org/10.64898/2026.03.19.712811.
Please cite this publication if you use the dataset in your research.
"""

import os
import zipfile
import tempfile
from typing import List, Tuple, Union

import numpy as np

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


BASE_URL = "https://zenodo.org/api/records/20513085/files"
RAW_ZIP_URL = f"{BASE_URL}/Raw_tomograms.zip/content"
LABEL_ZIP_URL = f"{BASE_URL}/Labels.zip/content"
METADATA_URL = f"{BASE_URL}/Cell_Metadata.csv/content"

LABEL_NAMES = {0: "exterior", 1: "cell", 2: "nucleus", 5: "mitochondria"}


def get_sxt_ins1e_mito_cell_names(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get the list of cell names in the SXT-INS1E-Mito dataset.

    Args:
        path: Filepath to a folder where the downloaded metadata will be saved.
        download: Whether to download the metadata if it is not present.

    Returns:
        The list of cell names.
    """
    os.makedirs(path, exist_ok=True)
    metadata_path = os.path.join(path, "Cell_Metadata.csv")
    util.download_source(metadata_path, METADATA_URL, download, checksum=None)

    with open(metadata_path) as f:
        next(f)  # Skip the header line.
        return [line.split(",")[0] for line in f if line.strip()]


def _read_mrc_member(zip_url, member_name):
    """Read one MRC member of a remote ZIP archive via HTTP range requests, without downloading the rest.

    `mrcfile` only accepts a filesystem path, so the member is written to a temporary file first.
    """
    import fsspec
    import mrcfile

    fs = fsspec.filesystem("http")
    with fs.open(zip_url, "rb") as f, zipfile.ZipFile(f) as zf:
        content = zf.read(member_name)

    with tempfile.NamedTemporaryFile(suffix=".mrc") as tmp:
        tmp.write(content)
        tmp.flush()
        with mrcfile.open(tmp.name, permissive=True) as mrc:
            return np.asarray(mrc.data)


def get_sxt_ins1e_mito_data(path: Union[os.PathLike, str], cell_name: str, download: bool = False) -> str:
    """Stream one cell's tomogram and label volume and cache it as a zarr v3 store.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        cell_name: The cell to fetch. See `get_sxt_ins1e_mito_cell_names`.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        The filepath to the cached zarr store.
    """
    import zarr
    from zarr.codecs import BloscCodec

    os.makedirs(path, exist_ok=True)
    zarr_path = os.path.join(path, f"{cell_name}.zarr")

    root = zarr.open_group(zarr_path, mode="a")
    if "raw" in root and "labels" in root:
        return zarr_path

    if not download:
        raise RuntimeError(f"No cached data found at '{zarr_path}'. Set download=True to stream it.")

    raw = _read_mrc_member(RAW_ZIP_URL, f"Raw_tomograms/{cell_name}_scaled.mrc")
    labels = _read_mrc_member(LABEL_ZIP_URL, f"Labels/{cell_name}_labels.mrc")

    assert raw.shape == labels.shape, f"Shape mismatch for '{cell_name}': {raw.shape} vs {labels.shape}"

    def _make_array(name, data, shuffle):
        array = root.create_array(
            name, shape=data.shape, chunks=(32, 256, 256), dtype=data.dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        array[:] = data

    root.attrs["cell_name"] = cell_name
    root.attrs["label_names"] = LABEL_NAMES

    _make_array("raw", raw, shuffle="shuffle")
    _make_array("labels", labels, shuffle="bitshuffle")

    return zarr_path


def get_sxt_ins1e_mito_paths(
    path: Union[os.PathLike, str], cell_names: List[str], download: bool = False,
) -> List[str]:
    """Get paths to cached SXT-INS1E-Mito zarr stores, one per cell.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        cell_names: Which cells to use. See `get_sxt_ins1e_mito_cell_names`.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    return [get_sxt_ins1e_mito_data(path, cell_name, download) for cell_name in cell_names]


def get_sxt_ins1e_mito_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    cell_names: List[str],
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the SXT-INS1E-Mito dataset for mitochondria and organelle segmentation.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        cell_names: Which cells to use. See `get_sxt_ins1e_mito_cell_names`.
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_sxt_ins1e_mito_paths(path, cell_names, download)
    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_sxt_ins1e_mito_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    cell_names: List[str],
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for mitochondria and organelle segmentation in the SXT-INS1E-Mito dataset.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        cell_names: Which cells to use. See `get_sxt_ins1e_mito_cell_names`.
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the
            PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_sxt_ins1e_mito_dataset(path, patch_shape, cell_names, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
