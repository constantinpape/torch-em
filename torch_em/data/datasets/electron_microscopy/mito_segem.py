"""The MitoSegEM dataset contains mitochondria instance segmentation for three large mouse SBF-SEM/
serial-section EM volumes: intestine, testis, and dorsal cochlear nucleus (brainstem).

Labels come from mito-segEM: a pretrained network prediction followed by webKnossos path-based
instance assignment. Annotation coverage is sparse, not exhaustive: some visible mitochondria in the
raw data are left unannotated.

The data is available at https://www.ebi.ac.uk/empiar/EMPIAR-12535/ under the CC0 license.
The dataset was published in https://doi.org/10.1016/j.crmeth.2025.100989.
Please cite this publication if you use the dataset in your research.
"""

import os
from io import BytesIO
from typing import Dict, List, Literal, Tuple, Union

import numpy as np

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


BASE_URL = "https://ftp.ebi.ac.uk/empiar/world_availability/12535/data"

TISSUES: Dict[str, dict] = {
    "intestine": {
        "raw_dir": "HuaLab-mouse-intestinal-tissue/raw",
        "seg_dir": "HuaLab-mouse-intestinal-tissue/intestinal_mito_segmentation_h5",
        "raw_pattern": "HuaLab-mouse-intestinal-tissue-raw-_{z:06d}.tiff",
        "first_slice_index": 1,
        "num_slices": 1000,
        "shape_yx": (5024, 6808),
        "resolution_nm": (50, 8, 8),
    },
    "testis": {
        "raw_dir": "HuaLab_mouse_testis_tissue/raw",
        "seg_dir": "HuaLab_mouse_testis_tissue/testis_mito_segmentation_h5",
        "raw_pattern": "HuaLab_mouse_testis_tissue-raw-_{z:06d}.tiff",
        "first_slice_index": 1,
        "num_slices": 2605,
        "shape_yx": (4884, 10419),
        "resolution_nm": (50, 15, 15),
    },
    "brainstem": {
        "raw_dir": "Hualab-DCN-A23-CBA-2M/raw",
        "seg_dir": "Hualab-DCN-A23-CBA-2M/brainstem_mito_segmentation_h5",
        "raw_pattern": "DUP_aligned_Prefix_ONPOINT_slice_{z:04d}.tif",
        "first_slice_index": 0,
        "num_slices": 3893,
        "shape_yx": (16449, 15303),
        "resolution_nm": (50, 15, 15),
    },
}


def _bbox_to_str(bounding_box):
    import hashlib
    return hashlib.md5("_".join(str(v) for v in bounding_box).encode()).hexdigest()[:12]


def _read_raw_slice(url, y_min, y_max, x_min, x_max):
    """Download one full raw slice and crop it to the requested region."""
    import time
    import requests
    import tifffile

    for attempt in range(5):
        try:
            response = requests.get(url, timeout=180)
            response.raise_for_status()
            image = tifffile.imread(BytesIO(response.content))
            return image[y_min:y_max, x_min:x_max]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def _read_label_slice(url, y_min, y_max, x_min, x_max):
    """Download one full label slice (h5) and crop it to the requested region."""
    import time
    import h5py
    import requests

    for attempt in range(5):
        try:
            response = requests.get(url, timeout=180)
            response.raise_for_status()
            with h5py.File(BytesIO(response.content), "r") as f:
                return f["data"][y_min:y_max, x_min:x_max]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def get_mito_segem_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[int, int, int, int, int, int],
    tissue: Literal["intestine", "testis", "brainstem"] = "intestine",
    download: bool = False,
) -> str:
    """Stream a subvolume of the MitoSegEM dataset and cache it as a zarr v3 store.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        bounding_box: The region to fetch as (z_min, z_max, y_min, y_max, x_min, x_max) in voxel
            coordinates, at each tissue's native resolution (see `TISSUES`).
        tissue: Which tissue volume to use. One of 'intestine', 'testis', 'brainstem'.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        The filepath to the cached zarr store.
    """
    import zarr
    from zarr.codecs import BloscCodec

    if tissue not in TISSUES:
        raise ValueError(f"tissue must be one of {list(TISSUES)}, got {tissue!r}")

    os.makedirs(str(path), exist_ok=True)
    zarr_path = os.path.join(str(path), f"{tissue}_{_bbox_to_str(bounding_box)}.zarr")

    root = zarr.open_group(zarr_path, mode="a")
    if "raw" in root and "labels" in root:
        return zarr_path

    if not download:
        raise RuntimeError(f"No cached data found at '{zarr_path}'. Set download=True to stream it.")

    z_min, z_max, y_min, y_max, x_min, x_max = bounding_box
    info = TISSUES[tissue]
    assert z_max <= info["num_slices"], f"z_max exceeds the {tissue} volume ({info['num_slices']} slices)"
    max_y, max_x = info["shape_yx"]
    assert y_max <= max_y and x_max <= max_x, f"Bounding box exceeds the {tissue} slice shape {info['shape_yx']}"

    shape = (z_max - z_min, y_max - y_min, x_max - x_min)
    raw = np.zeros(shape, dtype=np.uint8)
    labels = np.zeros(shape, dtype=np.uint32)

    for i, z in enumerate(range(z_min, z_max)):
        raw_url = f"{BASE_URL}/{info['raw_dir']}/{info['raw_pattern'].format(z=z + info['first_slice_index'])}"
        label_url = f"{BASE_URL}/{info['seg_dir']}/{z:04d}.h5"
        raw[i] = _read_raw_slice(raw_url, y_min, y_max, x_min, x_max)
        labels[i] = _read_label_slice(label_url, y_min, y_max, x_min, x_max)

    def _make_array(name, data, shuffle):
        array = root.create_array(
            name, shape=data.shape, chunks=(32, 512, 512), dtype=data.dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        array[:] = data

    root.attrs["bounding_box"] = list(bounding_box)
    root.attrs["tissue"] = tissue
    root.attrs["resolution_nm"] = list(info["resolution_nm"])
    root.attrs["labels_are_exhaustive"] = False

    _make_array("raw", raw, shuffle="shuffle")
    _make_array("labels", labels, shuffle="bitshuffle")

    return zarr_path


def get_mito_segem_paths(
    path: Union[os.PathLike, str],
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    tissue: Literal["intestine", "testis", "brainstem"] = "intestine",
    download: bool = False,
) -> List[str]:
    """Get paths to cached MitoSegEM zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        bounding_boxes: List of regions to fetch, each as (z_min, z_max, y_min, y_max, x_min, x_max).
        tissue: Which tissue volume to use. One of 'intestine', 'testis', 'brainstem'.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    return [get_mito_segem_data(path, bbox, tissue, download) for bbox in bounding_boxes]


def get_mito_segem_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    tissue: Literal["intestine", "testis", "brainstem"] = "intestine",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the MitoSegEM dataset for mitochondria instance segmentation.

    Labels are not exhaustive manual ground truth: some mitochondria may be unannotated.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_boxes: List of subvolumes to use, each as (z_min, z_max, y_min, y_max, x_min, x_max).
        tissue: Which tissue volume to use. One of 'intestine', 'testis', 'brainstem'.
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_mito_segem_paths(path, bounding_boxes, tissue, download)
    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_mito_segem_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    tissue: Literal["intestine", "testis", "brainstem"] = "intestine",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for mitochondria instance segmentation in the MitoSegEM dataset.

    Labels are not exhaustive manual ground truth: some mitochondria may be unannotated.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        bounding_boxes: List of subvolumes to use, each as (z_min, z_max, y_min, y_max, x_min, x_max).
        tissue: Which tissue volume to use. One of 'intestine', 'testis', 'brainstem'.
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the
            PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mito_segem_dataset(path, patch_shape, bounding_boxes, tissue, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
