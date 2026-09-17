"""The FluoEM dataset contains an electron microscopy volume of mouse cortex with proofread
axon instance segmentation, generated to validate a fluorescence-guided axon labeling method
(FluoEM) for long-range connectomics.

Segmentation is dense but not uniformly proofread throughout the volume: some regions still show
block-stitching seams from the automated reconstruction, visible as brief instance ID mismatches
at internal block boundaries.

The data is available via the Helmstaedter Lab's public WEBKNOSSOS instance (organization
MPI_Brain_Research). No license is specified by the data provider.
The dataset was published in https://doi.org/10.7554/eLife.38976.
Please cite this publication if you use the dataset in your research.
"""

import os
from typing import Tuple, List, Union

import numpy as np

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


DATASTORE_URL = "https://demo.wk1.connectomics.hpccloud.mpg.de/data/zarr"
DATASET_ID = "5abb89f348d7a73cea448019"
SHAPE_ZYX = (2767, 26242, 29000)
RESOLUTION_NM = (30, 11.24, 11.24)


def _bbox_to_str(bounding_box):
    import hashlib
    return hashlib.md5("_".join(str(v) for v in bounding_box).encode()).hexdigest()[:12]


def _open_remote_array(layer):
    import fsspec
    import zarr
    return zarr.open(fsspec.get_mapper(f"{DATASTORE_URL}/{DATASET_ID}/{layer}/1-1-1"), mode="r", zarr_format=2)


def get_fluoem_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[int, int, int, int, int, int],
    download: bool = False,
) -> str:
    """Stream a subvolume of the FluoEM dataset and cache it as a zarr v3 store.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        bounding_box: The region to fetch as (z_min, z_max, y_min, y_max, x_min, x_max) in voxel
            coordinates, at the dataset's native resolution.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        The filepath to the cached zarr store.
    """
    import zarr
    from zarr.codecs import BloscCodec

    os.makedirs(str(path), exist_ok=True)
    zarr_path = os.path.join(str(path), f"{_bbox_to_str(bounding_box)}.zarr")

    root = zarr.open_group(zarr_path, mode="a")
    if "raw" in root and "labels" in root:
        return zarr_path

    if not download:
        raise RuntimeError(f"No cached data found at '{zarr_path}'. Set download=True to stream it.")

    z_min, z_max, y_min, y_max, x_min, x_max = bounding_box
    shape_z, shape_y, shape_x = SHAPE_ZYX
    assert z_max <= shape_z and y_max <= shape_y and x_max <= shape_x, \
        f"Bounding box exceeds the FluoEM volume {SHAPE_ZYX}"

    color = _open_remote_array("color")
    seg = _open_remote_array("segmentation")

    # the remote arrays are indexed as (channel, x, y, z); torch_em volumes use (z, y, x).
    raw = np.transpose(color[0, x_min:x_max, y_min:y_max, z_min:z_max], (2, 1, 0))
    labels = np.transpose(seg[0, x_min:x_max, y_min:y_max, z_min:z_max], (2, 1, 0))

    def _make_array(name, data, shuffle):
        array = root.create_array(
            name, shape=data.shape, chunks=(32, 512, 512), dtype=data.dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        array[:] = data

    root.attrs["bounding_box"] = list(bounding_box)
    root.attrs["resolution_nm"] = list(RESOLUTION_NM)
    root.attrs["labels_are_exhaustive"] = False

    _make_array("raw", raw, shuffle="shuffle")
    _make_array("labels", labels, shuffle="bitshuffle")

    return zarr_path


def get_fluoem_paths(
    path: Union[os.PathLike, str],
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    download: bool = False,
) -> List[str]:
    """Get paths to cached FluoEM zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        bounding_boxes: List of regions to fetch, each as (z_min, z_max, y_min, y_max, x_min, x_max).
        download: Whether to stream and cache the data if it is not present.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    return [get_fluoem_data(path, bbox, download) for bbox in bounding_boxes]


def get_fluoem_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the FluoEM dataset for axon instance segmentation.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_boxes: List of subvolumes to use, each as (z_min, z_max, y_min, y_max, x_min, x_max).
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_fluoem_paths(path, bounding_boxes, download)
    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_fluoem_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for axon instance segmentation in the FluoEM dataset.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        bounding_boxes: List of subvolumes to use, each as (z_min, z_max, y_min, y_max, x_min, x_max).
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the
            PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_fluoem_dataset(path, patch_shape, bounding_boxes, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
