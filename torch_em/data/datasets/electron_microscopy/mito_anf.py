"""The Mito-ANFs dataset contains mitochondria instance segmentation for two mouse cochlear
serial block-face EM (SBEM) volumes of auditory nerve fibers in the inner spiral bundle region,
at 12x12x50 nm (XYZ) resolution.

Labels come from a 3D U-Net prediction followed by manual proofreading. Annotation coverage is
sparse, not exhaustive: only mitochondria belonging to traced auditory nerve fibers are labeled.

The data is available at https://webknossos.org/datasets/b2275d664e4c2a96 (organization
'Yunfeng Hua Lab'). No license is specified by the data provider.
The dataset was published in https://doi.org/10.1007/s10162-024-00957-y.
Please cite this publication if you use the dataset in your research.
"""

import os
from typing import Dict, List, Literal, Tuple, Union

import numpy as np

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


BASE_URL = "https://data-humerus.webknossos.org/data/zarr"

SAMPLES: Dict[str, dict] = {
    "M1": {
        "dataset_id": "652d442501000053049c0270",
        "seg_layer": "mito_0_890_proof",
        "shape_zyx": (2494, 23573, 11333),
        "resolution_nm": (50, 12, 12),
    },
    "M2": {
        "dataset_id": "652d563301000068049c066e",
        "seg_layer": "mito_650_1849_proof",
        "shape_zyx": (2558, 25183, 10329),
        "resolution_nm": (50, 12, 12),
    },
}


def _bbox_to_str(bounding_box):
    import hashlib
    return hashlib.md5("_".join(str(v) for v in bounding_box).encode()).hexdigest()[:12]


def _open_remote_array(dataset_id, layer):
    import fsspec
    import zarr
    return zarr.open(fsspec.get_mapper(f"{BASE_URL}/{dataset_id}/{layer}/1-1-1"), mode="r", zarr_format=2)


def get_mito_anf_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[int, int, int, int, int, int],
    sample: Literal["M1", "M2"] = "M1",
    download: bool = False,
) -> str:
    """Stream a subvolume of the Mito-ANFs dataset and cache it as a zarr v3 store.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        bounding_box: The region to fetch as (z_min, z_max, y_min, y_max, x_min, x_max) in voxel
            coordinates, at the dataset's native resolution (see `SAMPLES`).
        sample: Which cochlear sample to use. One of 'M1', 'M2'.
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

    color = _open_remote_array(info["dataset_id"], "color")
    seg = _open_remote_array(info["dataset_id"], info["seg_layer"])

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


def get_mito_anf_paths(
    path: Union[os.PathLike, str],
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    sample: Literal["M1", "M2"] = "M1",
    download: bool = False,
) -> List[str]:
    """Get paths to cached Mito-ANFs zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        bounding_boxes: List of regions to fetch, each as (z_min, z_max, y_min, y_max, x_min, x_max).
        sample: Which cochlear sample to use. One of 'M1', 'M2'.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    return [get_mito_anf_data(path, bbox, sample, download) for bbox in bounding_boxes]


def get_mito_anf_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    sample: Literal["M1", "M2"] = "M1",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Mito-ANFs dataset for mitochondria instance segmentation.

    Labels are not exhaustive manual ground truth: only mitochondria belonging to traced
    auditory nerve fibers are annotated.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_boxes: List of subvolumes to use, each as (z_min, z_max, y_min, y_max, x_min, x_max).
        sample: Which cochlear sample to use. One of 'M1', 'M2'.
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_mito_anf_paths(path, bounding_boxes, sample, download)
    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_mito_anf_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    sample: Literal["M1", "M2"] = "M1",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for mitochondria instance segmentation in the Mito-ANFs dataset.

    Labels are not exhaustive manual ground truth: only mitochondria belonging to traced
    auditory nerve fibers are annotated.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        bounding_boxes: List of subvolumes to use, each as (z_min, z_max, y_min, y_max, x_min, x_max).
        sample: Which cochlear sample to use. One of 'M1', 'M2'.
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the
            PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mito_anf_dataset(path, patch_shape, bounding_boxes, sample, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
