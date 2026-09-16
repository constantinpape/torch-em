"""The CortexConnectomics dataset contains dense electron microscopy connectomics volumes of
cortex from mouse, macaque, and human, with proofread neuron instance segmentation, acquired to
compare synaptic circuit architecture across species.

Nine of the twelve datasets published with the paper include a proofread segmentation layer; the
remaining three (two human MultiSEM volumes and one extended mouse posterior parietal cortex
volume) provide only raw EM data and are not included here.

Segmentation is dense but not uniformly proofread throughout each volume: some regions still show
block-stitching seams from the automated reconstruction, visible as brief instance ID mismatches
at internal block boundaries.

The data is available at https://demo.wk1.connectomics.hpccloud.mpg.de (organization
'Helmstaedter Lab', Max Planck Institute for Brain Research). No license is specified by the data
provider.
The dataset was published in https://doi.org/10.1126/science.abo0924.
Please cite this publication if you use the dataset in your research.
"""

import os
from typing import Dict, List, Literal, Tuple, Union

import numpy as np

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


BASE_URL = "https://demo.wk1.connectomics.hpccloud.mpg.de/data/zarr"

SAMPLES: Dict[str, dict] = {
    "mouse_s1": {
        "dataset_id": "62b19325010000860033e7c9", "shape_zyx": (7630, 5030, 7874),
        "resolution_nm": (28, 11.24, 11.24),
    },
    "mouse_ppc": {
        "dataset_id": "5c1cd439010000e66f8c2786", "shape_zyx": (4864, 12032, 12032), "resolution_nm": (30, 12, 12),
    },
    "mouse_acc": {
        "dataset_id": "5c1cd43801000026708c2783", "shape_zyx": (3328, 14336, 9216), "resolution_nm": (30, 12, 12),
    },
    "mouse_v2": {
        "dataset_id": "5c1cd4380100004f6f8c2785", "shape_zyx": (5120, 10240, 7168), "resolution_nm": (30, 12, 12),
    },
    "mouse_a2": {
        "dataset_id": "62b17edd010000f20075d7b9", "shape_zyx": (3642, 16102, 10394),
        "resolution_nm": (30, 11.24, 11.24),
    },
    "macaque_s1": {
        "dataset_id": "62b17f19010000aa0075d7bb", "shape_zyx": (3359, 19544, 14942),
        "resolution_nm": (30, 11.24, 11.24),
    },
    "macaque_stg": {
        "dataset_id": "62b17f55010000eb0075d7be", "shape_zyx": (3599, 20324, 15907),
        "resolution_nm": (30, 11.24, 11.24),
    },
    "human_stg": {
        "dataset_id": "62b17f19010000aa0075d7bc", "shape_zyx": (3759, 19260, 14845),
        "resolution_nm": (30, 11.24, 11.24),
    },
    "human_ifg": {
        "dataset_id": "62b17fcd010000eb0075d7c0", "shape_zyx": (2626, 19197, 15110),
        "resolution_nm": (30, 11.24, 11.24),
    },
}


def _bbox_to_str(bounding_box):
    import hashlib
    return hashlib.md5("_".join(str(v) for v in bounding_box).encode()).hexdigest()[:12]


def _open_remote_array(dataset_id, layer):
    import fsspec
    import zarr
    return zarr.open(fsspec.get_mapper(f"{BASE_URL}/{dataset_id}/{layer}/1-1-1"), mode="r", zarr_format=2)


def get_cortex_connectomics_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[int, int, int, int, int, int],
    sample: Literal[
        "mouse_s1", "mouse_ppc", "mouse_acc", "mouse_v2", "mouse_a2", "macaque_s1", "macaque_stg",
        "human_stg", "human_ifg",
    ] = "mouse_s1",
    download: bool = False,
) -> str:
    """Stream a subvolume of the CortexConnectomics dataset and cache it as a zarr v3 store.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        bounding_box: The region to fetch as (z_min, z_max, y_min, y_max, x_min, x_max) in voxel
            coordinates, at the dataset's native resolution (see `SAMPLES`).
        sample: Which species / cortical region to use. See `SAMPLES` for the available keys.
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
    seg = _open_remote_array(info["dataset_id"], "segmentation")

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


def get_cortex_connectomics_paths(
    path: Union[os.PathLike, str],
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    sample: Literal[
        "mouse_s1", "mouse_ppc", "mouse_acc", "mouse_v2", "mouse_a2", "macaque_s1", "macaque_stg",
        "human_stg", "human_ifg",
    ] = "mouse_s1",
    download: bool = False,
) -> List[str]:
    """Get paths to cached CortexConnectomics zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        bounding_boxes: List of regions to fetch, each as (z_min, z_max, y_min, y_max, x_min, x_max).
        sample: Which species / cortical region to use. See `SAMPLES` for the available keys.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    return [get_cortex_connectomics_data(path, bbox, sample, download) for bbox in bounding_boxes]


def get_cortex_connectomics_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    sample: Literal[
        "mouse_s1", "mouse_ppc", "mouse_acc", "mouse_v2", "mouse_a2", "macaque_s1", "macaque_stg",
        "human_stg", "human_ifg",
    ] = "mouse_s1",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the CortexConnectomics dataset for neuron instance segmentation.

    Labels are dense within the proofread region of each volume, but that region can be smaller
    than the full raw volume; choose bounding boxes accordingly.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_boxes: List of subvolumes to use, each as (z_min, z_max, y_min, y_max, x_min, x_max).
        sample: Which species / cortical region to use. See `SAMPLES` for the available keys.
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_cortex_connectomics_paths(path, bounding_boxes, sample, download)
    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_cortex_connectomics_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    sample: Literal[
        "mouse_s1", "mouse_ppc", "mouse_acc", "mouse_v2", "mouse_a2", "macaque_s1", "macaque_stg",
        "human_stg", "human_ifg",
    ] = "mouse_s1",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for neuron instance segmentation in the CortexConnectomics dataset.

    Labels are dense within the proofread region of each volume, but that region can be smaller
    than the full raw volume; choose bounding boxes accordingly.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        bounding_boxes: List of subvolumes to use, each as (z_min, z_max, y_min, y_max, x_min, x_max).
        sample: Which species / cortical region to use. See `SAMPLES` for the available keys.
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the
            PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cortex_connectomics_dataset(path, patch_shape, bounding_boxes, sample, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
