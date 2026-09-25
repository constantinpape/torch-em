"""The FAFB (Full Adult Fly Brain) dataset contains a serial-section TEM volume of the
full adult female Drosophila brain with dense neuron instance segmentation from FlyWire.

The EM (FAFB v14) is a ssTEM dataset. The native 4 x 4 x 40 nm mip level is a
placeholder with no data and mip=1 (8 x 8 x 40 nm) holds EM only. mip=2 (16 x 16 x 40 nm)
is the finest level of the FlyWire neuron segmentation (materialization v783, Nature 2024
paper), so both are used at 16 x 16 x 40 nm.

Bounding boxes are specified in 16 x 16 x 40 nm voxel coordinates
(x_min, x_max, y_min, y_max, z_min, z_max).
Valid coordinate overlap between EM (mip=2) and seg: x=[5100,59200], y=[1440,29600], z=[16,7062].

The EM is at gs://microns-seunglab/drosophila_v0/alignment/image_rechunked (mip=2) and
the neuron segmentation (v783) is at gs://flywire_v141_m783.

This dataset is from the publication https://doi.org/10.1038/s41586-024-07558-y.
Please cite it if you use this dataset in your research.

The dataset is publicly available at https://flywire.ai.
Requires cloud-volume: pip install cloud-volume.

NOTE (on data size): the full seg volume is (54100, 28160, 7046) voxels at 16 x 16 x 40 nm.
Downloading the entire volume is not feasible. Data is streamed from GCS and cached
locally as zarr v3 stores by specifying bounding boxes.

NOTE (AA): The data annotations are amazing, I personally think that the segmentation
resolution is too low. If we wanna use it, we should go one resolution higher
(we are at s2 atm).
"""

import hashlib
import os
from typing import List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch_em
from .. import util


EM_URL = "gs://microns-seunglab/drosophila_v0/alignment/image_rechunked"
SEG_URL = "gs://flywire_v141_m783"
# mip=2 gives 16x16x40nm, matching the seg resolution; mip=0 is a placeholder with no data.
EM_MIP = 2

# 1024x1024x410-voxel crops (16 x 16 x 16 um) inside brain tissue at three depths; the brain fills only part of
# the coordinate range, so a box must be checked against the data.
DEFAULT_BOUNDING_BOXES = [
    (35840, 36864, 6656, 7680, 1500, 1910),  # right, dorsal, anterior
    (32768, 33792, 18944, 19968, 1500, 1910),  # midline, ventral, anterior
    (15360, 16384, 17920, 18944, 3500, 3910),  # left optic lobe, ventral, mid-depth
    (48128, 49152, 18944, 19968, 3500, 3910),  # right optic lobe, ventral, mid-depth
    (24576, 25600, 11776, 12800, 3500, 3910),  # left, dorsal, mid-depth
    (41984, 43008, 12800, 13824, 3500, 3910),  # right, mid-height, mid-depth
    (18432, 19456, 11776, 12800, 5500, 5910),  # left, dorsal, posterior
    (21504, 22528, 17920, 18944, 5500, 5910),  # left, ventral, posterior
    (23552, 24576, 14848, 15872, 5500, 5910),  # left, mid-height, posterior
]
DEFAULT_BOUNDING_BOX = DEFAULT_BOUNDING_BOXES[4]

FAFB_CHUNK_SHAPE = (64, 256, 256)


def _bbox_to_str(bbox):
    return hashlib.md5("_".join(str(v) for v in bbox).encode()).hexdigest()[:12]


def _create_array(root, name, shape, dtype, is_label):
    from zarr.codecs import BloscCodec
    shuffle = "bitshuffle" if (np.issubdtype(dtype, np.integer) and is_label) else "shuffle"
    return root.create_array(
        name,
        shape=shape,
        chunks=FAFB_CHUNK_SHAPE,
        dtype=dtype,
        compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
    )


def get_fafb_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[int, int, int, int, int, int] = DEFAULT_BOUNDING_BOX,
    download: bool = False,
) -> str:
    """Stream a subvolume from the FAFB dataset and cache it as a zarr v3 store.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        bounding_box: The region to fetch as (x_min, x_max, y_min, y_max, z_min, z_max)
            in 16 nm voxel coordinates. Defaults to DEFAULT_BOUNDING_BOXES, 1024x1024x410 crops inside brain tissue.
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
    print(f"Streaming FAFB EM + FlyWire segmentation for bbox {bounding_box} ...")

    em_vol = cloudvolume.CloudVolume(EM_URL, use_https=True, mip=EM_MIP, progress=True)
    seg_vol = cloudvolume.CloudVolume(SEG_URL, use_https=True, mip=0, progress=True)

    raw = np.array(em_vol[x_min:x_max, y_min:y_max, z_min:z_max])[..., 0].transpose(2, 1, 0)
    labels = np.array(seg_vol[x_min:x_max, y_min:y_max, z_min:z_max])[..., 0].transpose(2, 1, 0)

    # The brain fills only part of the coordinate range and the servers return zeros outside it.
    if not raw.any() or len(np.unique(labels)) < 2:
        raise RuntimeError(
            f"The bounding box {bounding_box} holds no tissue or no segmentation. "
            "Pick a box inside the brain, e.g. one of DEFAULT_BOUNDING_BOXES."
        )

    # FlyWire IDs are large uint64 values - relabel to consecutive integers.
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.reshape(raw.shape).astype("uint64")

    shape = tuple(min(r, l) for r, l in zip(raw.shape, labels.shape))
    raw = raw[:shape[0], :shape[1], :shape[2]]
    labels = labels[:shape[0], :shape[1], :shape[2]]

    root.attrs["bounding_box"] = list(bounding_box)
    root.attrs["resolution_nm"] = [16, 16, 40]

    if "raw" not in root:
        ds_raw = _create_array(root, "raw", shape, np.dtype("uint8"), is_label=False)
        ds_raw[:] = raw
    if "labels" not in root:
        ds_lbl = _create_array(root, "labels", shape, np.dtype("uint64"), is_label=True)
        ds_lbl[:] = labels

    print(f"Cached to {zarr_path} (shape {shape})")
    return zarr_path


def get_fafb_paths(
    path: Union[os.PathLike, str],
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to FAFB zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        bounding_boxes: List of regions to fetch, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 16 nm voxel coordinates.
            Defaults to DEFAULT_BOUNDING_BOXES, 1024x1024x410 crops inside brain tissue.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    if bounding_boxes is None:
        bounding_boxes = DEFAULT_BOUNDING_BOXES
    return [get_fafb_data(path, bbox, download) for bbox in bounding_boxes]


def get_fafb_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the FAFB dataset for neuron instance segmentation.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 16 nm voxel coordinates.
            Defaults to DEFAULT_BOUNDING_BOXES, 1024x1024x410 crops inside brain tissue.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_fafb_paths(path, bounding_boxes, download)

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


def get_fafb_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for neuron instance segmentation in the FAFB dataset.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 16 nm voxel coordinates.
            Defaults to DEFAULT_BOUNDING_BOXES, 1024x1024x410 crops inside brain tissue.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_fafb_dataset(
        path, patch_shape, bounding_boxes=bounding_boxes,
        download=download, offsets=offsets, boundaries=boundaries, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
