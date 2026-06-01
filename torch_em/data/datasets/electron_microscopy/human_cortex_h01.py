"""The H01 dataset contains a petavoxel fragment of human cerebral cortex
reconstructed at nanoscale resolution.

The dataset covers approximately 1 mm³ of human temporal lobe acquired with
ssTEM at 4 x 4 x 33 nm native resolution. The neuron segmentation (c3 agglomeration)
is at 8 x 8 x 33 nm with ~50,000 reconstructed cells. The EM is accessed at mip=1
(8 x 8 x 33 nm) to match the segmentation resolution.

Bounding boxes are specified in 8 x 8 x 33 nm segmentation voxel coordinates
(x_min, x_max, y_min, y_max, z_min, z_max). The full volume is (515892, 356400, 5293)
voxels at this resolution.

The EM is at gs://h01-release/data/20210601/4nm_raw and the segmentation (c3) is at
gs://h01-release/data/20210601/c3. Both require only public GCS access.

This dataset is from the publication https://doi.org/10.1126/science.adk4858.
Please cite it if you use this dataset in your research.

The dataset is publicly available at https://h01-release.storage.googleapis.com.
Requires cloud-volume: pip install cloud-volume.

NOTE (on annotations): the c3 segmentation is sparse - ~70-78% of voxels are labeled
in tissue regions, with the remainder being extracellular space or unassigned processes.
Use MinInstanceSampler to avoid empty patches during training.

NOTE (on data size): the full volume is (515892, 356400, 5293) voxels at 8 x 8 x 33 nm.
Data is streamed from GCS and cached locally as zarr v3 stores by specifying bounding boxes.
"""

import hashlib
import os
from typing import List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch_em
from .. import util


EM_URL = "gs://h01-release/data/20210601/4nm_raw"
SEG_URL = "gs://h01-release/data/20210601/c3"
# EM mip=1 gives 8x8x33nm, matching the seg resolution at mip=0.
EM_MIP = 1

# Four 2048x2048x512-voxel crops sampling different cortical regions.
# At 8x8x33 nm this gives ~16.4x16.4x16.9 um physically isotropic subvolumes.
# Full volume: (515892, 356400, 5293) voxels at 8x8x33 nm.
DEFAULT_BOUNDING_BOXES = [
    (50000, 52048, 50000, 52048, 500, 1012),       # lower-left, low z
    (257000, 259048, 178000, 180048, 2500, 3012),  # central
    (460000, 462048, 300000, 302048, 4500, 5012),  # upper-right, high z
    (150000, 152048, 250000, 252048, 3500, 4012),  # mid-volume
]
DEFAULT_BOUNDING_BOX = DEFAULT_BOUNDING_BOXES[1]

HUMAN_CORTEX_H01_CHUNK_SHAPE = (64, 256, 256)


def _bbox_to_str(bbox):
    return hashlib.md5("_".join(str(v) for v in bbox).encode()).hexdigest()[:12]


def _create_array(root, name, shape, dtype, is_label):
    from zarr.codecs import BloscCodec
    shuffle = "bitshuffle" if (np.issubdtype(dtype, np.integer) and is_label) else "shuffle"
    return root.create_array(
        name,
        shape=shape,
        chunks=HUMAN_CORTEX_H01_CHUNK_SHAPE,
        dtype=dtype,
        compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
    )


def get_human_cortex_h01_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[int, int, int, int, int, int] = DEFAULT_BOUNDING_BOX,
    download: bool = False,
) -> str:
    """Stream a subvolume from the H01 dataset and cache it as a zarr v3 store.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        bounding_box: The region to fetch as (x_min, x_max, y_min, y_max, z_min, z_max)
            in 8 x 8 x 33 nm voxel coordinates. Defaults to a 2048x2048x512 central crop.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        The filepath to the cached zarr store.
    """
    import zarr

    os.makedirs(str(path), exist_ok=True)
    zarr_path = os.path.join(str(path), f"{_bbox_to_str(bounding_box)}.zarr")

    root = zarr.open_group(zarr_path, mode="a")
    if "raw" in root and "labels" in root:
        return zarr_path

    if not download:
        raise RuntimeError(
            f"No cached data found at '{zarr_path}'. Set download=True to stream it from GCS."
        )

    try:
        import cloudvolume
    except ImportError:
        raise ImportError("The 'cloud-volume' package is required: pip install cloud-volume")

    x_min, x_max, y_min, y_max, z_min, z_max = bounding_box
    print(f"Streaming H01 EM + segmentation for bbox {bounding_box} ...")

    em_vol = cloudvolume.CloudVolume(EM_URL, use_https=True, mip=EM_MIP, progress=True)
    seg_vol = cloudvolume.CloudVolume(SEG_URL, use_https=True, mip=0, progress=True)

    raw = np.array(em_vol[x_min:x_max, y_min:y_max, z_min:z_max])[..., 0].transpose(2, 1, 0)
    labels = np.array(seg_vol[x_min:x_max, y_min:y_max, z_min:z_max])[..., 0].transpose(2, 1, 0)

    # H01 IDs are large uint64 values - relabel to consecutive integers.
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.reshape(raw.shape).astype("uint64")

    shape = tuple(min(r, l) for r, l in zip(raw.shape, labels.shape))
    raw = raw[:shape[0], :shape[1], :shape[2]]
    labels = labels[:shape[0], :shape[1], :shape[2]]

    root.attrs["bounding_box"] = list(bounding_box)
    root.attrs["resolution_nm"] = [8, 8, 33]

    if "raw" not in root:
        ds_raw = _create_array(root, "raw", shape, np.dtype("uint8"), is_label=False)
        ds_raw[:] = raw
    if "labels" not in root:
        ds_lbl = _create_array(root, "labels", shape, np.dtype("uint64"), is_label=True)
        ds_lbl[:] = labels

    print(f"Cached to {zarr_path} (shape {shape})")
    return zarr_path


def get_human_cortex_h01_paths(
    path: Union[os.PathLike, str],
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to H01 zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        bounding_boxes: List of regions to fetch, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 x 8 x 33 nm voxel coordinates.
            Defaults to DEFAULT_BOUNDING_BOXES (4 crops).
        download: Whether to stream and cache the data if it is not present.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    if bounding_boxes is None:
        bounding_boxes = DEFAULT_BOUNDING_BOXES
    return [get_human_cortex_h01_data(path, bbox, download) for bbox in bounding_boxes]


def get_human_cortex_h01_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the H01 dataset for neuron instance segmentation.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 x 8 x 33 nm voxel coordinates.
            Defaults to DEFAULT_BOUNDING_BOXES - four 2048x2048x512 isotropic crops.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_human_cortex_h01_paths(path, bounding_boxes, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_human_cortex_h01_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for neuron instance segmentation in the H01 dataset.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 x 8 x 33 nm voxel coordinates.
            Defaults to DEFAULT_BOUNDING_BOXES - four 2048x2048x512 isotropic crops.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_human_cortex_h01_dataset(
        path, patch_shape, bounding_boxes=bounding_boxes, download=download,
        offsets=offsets, boundaries=boundaries, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
