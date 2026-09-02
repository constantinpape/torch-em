"""The FAFB (Full Adult Fly Brain) dataset contains a serial-section TEM volume of the
full adult female Drosophila brain with dense neuron instance segmentation from FlyWire.

The EM (FAFB v14) is a ssTEM dataset. The native 4 x 4 x 40 nm mip level is a
placeholder with no data - the finest available EM is at mip=2 (16 x 16 x 40 nm),
which matches the FlyWire neuron segmentation (materialization v783, Nature 2024 paper)
resolution exactly. Both are stored at 16 x 16 x 40 nm.

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

# Four 2048x2048x819-voxel crops sampling different brain regions.
# At 16x16x40 nm this gives ~32x32x32 um physically isotropic subvolumes.
DEFAULT_BOUNDING_BOXES = [
    (6000, 8048, 2000, 4048, 500, 1319),  # anterior-left, low z
    (31000, 33048, 14500, 16548, 3200, 4019),  # central brain
    (56000, 58048, 26500, 28548, 5800, 6619),  # posterior-right, high z
    (15000, 17048, 8000, 10048, 6100, 6919),  # mid-left, high z
]
DEFAULT_BOUNDING_BOX = DEFAULT_BOUNDING_BOXES[1]

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
            in 16 nm voxel coordinates. Defaults to a 2048x2048x819 central brain crop.
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
            Defaults to DEFAULT_BOUNDING_BOXES (4 crops).
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
            Defaults to DEFAULT_BOUNDING_BOXES - four 2048x2048x819 isotropic crops.
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
            Defaults to DEFAULT_BOUNDING_BOXES - four 2048x2048x819 isotropic crops.
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
