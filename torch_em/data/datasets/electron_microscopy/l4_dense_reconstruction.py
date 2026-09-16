"""The L4Dense dataset contains a dense electron microscopy connectomic reconstruction of layer 4
of mouse somatosensory (barrel) cortex, with proofread neuron instance segmentation.

Two samples are available: the full reconstructed volume ('full') and a smaller public demo
subvolume ('demo').

Segmentation is dense but not uniformly proofread throughout each volume: some regions still show
block-stitching seams from the automated reconstruction, visible as brief instance ID mismatches
at internal block boundaries.

The data is available via the Helmstaedter Lab's public WEBKNOSSOS instances (organization
MPI_Brain_Research); the paper's own data repository at https://l4dense2019.brain.mpg.de was
unreachable at the time of writing this module. No license is specified by the data provider.
The dataset was published in https://doi.org/10.1126/science.aay3134.
Please cite this publication if you use the dataset in your research.
"""

import os
from typing import Dict, List, Literal, Tuple, Union

import numpy as np

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


SAMPLES: Dict[str, dict] = {
    "full": {
        "datastore_url": "https://demo.wk1.connectomics.hpccloud.mpg.de/data/zarr",
        "dataset_id": "5da84e9401000018006d210b",
        "shape_zyx": (3424, 8534, 5599),
        "resolution_nm": (28, 11.24, 11.24),
    },
    "demo": {
        "datastore_url": "https://data-humerus.webknossos.org/data/zarr",
        "dataset_id": "5a7c737448d7a73cea3c83a8",
        "shape_zyx": (714, 1779, 1779),
        "resolution_nm": (28, 11.24, 11.24),
    },
}


def _bbox_to_str(bounding_box):
    import hashlib
    return hashlib.md5("_".join(str(v) for v in bounding_box).encode()).hexdigest()[:12]


def _open_remote_array(datastore_url, dataset_id, layer):
    import fsspec
    import zarr
    return zarr.open(fsspec.get_mapper(f"{datastore_url}/{dataset_id}/{layer}/1-1-1"), mode="r", zarr_format=2)


def get_l4_dense_reconstruction_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[int, int, int, int, int, int],
    sample: Literal["full", "demo"] = "full",
    download: bool = False,
) -> str:
    """Stream a subvolume of the L4Dense dataset and cache it as a zarr v3 store.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        bounding_box: The region to fetch as (z_min, z_max, y_min, y_max, x_min, x_max) in voxel
            coordinates, at the dataset's native resolution (see `SAMPLES`).
        sample: Which sample to use, 'full' or 'demo'.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        The filepath to the cached zarr store.
    """
    import zarr
    from zarr.codecs import BloscCodec

    if sample not in SAMPLES:
        raise ValueError(f"sample must be one of {list(SAMPLES)}, got {sample!r}")

    os.makedirs(str(path), exist_ok=True)
    zarr_path = os.path.join(str(path), f"{sample}_{_bbox_to_str(bounding_box)}.zarr")

    root = zarr.open_group(zarr_path, mode="a")
    if "raw" in root and "labels" in root:
        return zarr_path

    if not download:
        raise RuntimeError(f"No cached data found at '{zarr_path}'. Set download=True to stream it.")

    z_min, z_max, y_min, y_max, x_min, x_max = bounding_box
    info = SAMPLES[sample]
    shape_z, shape_y, shape_x = info["shape_zyx"]
    assert z_max <= shape_z and y_max <= shape_y and x_max <= shape_x, \
        f"Bounding box exceeds the {sample} volume {info['shape_zyx']}"

    color = _open_remote_array(info["datastore_url"], info["dataset_id"], "color")
    seg = _open_remote_array(info["datastore_url"], info["dataset_id"], "segmentation")

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
    root.attrs["sample"] = sample
    root.attrs["resolution_nm"] = list(info["resolution_nm"])
    root.attrs["labels_are_exhaustive"] = False

    _make_array("raw", raw, shuffle="shuffle")
    _make_array("labels", labels, shuffle="bitshuffle")

    return zarr_path


def get_l4_dense_reconstruction_paths(
    path: Union[os.PathLike, str],
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    sample: Literal["full", "demo"] = "full",
    download: bool = False,
) -> List[str]:
    """Get paths to cached L4Dense zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        bounding_boxes: List of regions to fetch, each as (z_min, z_max, y_min, y_max, x_min, x_max).
        sample: Which sample to use, 'full' or 'demo'.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    return [get_l4_dense_reconstruction_data(path, bbox, sample, download) for bbox in bounding_boxes]


def get_l4_dense_reconstruction_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    sample: Literal["full", "demo"] = "full",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the L4Dense dataset for neuron instance segmentation.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_boxes: List of subvolumes to use, each as (z_min, z_max, y_min, y_max, x_min, x_max).
        sample: Which sample to use, 'full' or 'demo'.
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_l4_dense_reconstruction_paths(path, bounding_boxes, sample, download)
    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_l4_dense_reconstruction_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    sample: Literal["full", "demo"] = "full",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for neuron instance segmentation in the L4Dense dataset.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        bounding_boxes: List of subvolumes to use, each as (z_min, z_max, y_min, y_max, x_min, x_max).
        sample: Which sample to use, 'full' or 'demo'.
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the
            PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_l4_dense_reconstruction_dataset(path, patch_shape, bounding_boxes, sample, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
