"""The MANC (Male Adult Nerve Cord) dataset contains a FIB-SEM volume of the
Drosophila male ventral nerve cord with dense neuron instance segmentation.

It covers the full adult male nerve cord at 8 nm isotropic resolution with
~23,000 neurons reconstructed and proofread, including 10 million pre-synaptic
sites and 74 million post-synaptic densities.

The EM volume is at gs://flyem-vnc-2-26-213dba213ef26e094c16c860ae7f4be0/v3_emdata_clahe_xy/jpeg
and the segmentation is at gs://manc-seg-v1p2/manc-seg-v1.2.

This dataset is from the publication https://doi.org/10.7554/eLife.89346.
Please cite it if you use this dataset in your research.

The dataset is publicly available at https://www.janelia.org/project-team/flyem/manc-connectome.
Requires cloud-volume: pip install cloud-volume.

NOTE (on data size): the full volume is (46113, 59467, 82276) voxels at 8 nm isotropic
resolution. Downloading the entire volume is not feasible. Data is instead accessed by
specifying bounding boxes (x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel
coordinates, streamed from GCS and cached locally as HDF5 files.
"""

import hashlib
import os
from typing import List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch_em
from .. import util


EM_URL = "gs://flyem-vnc-2-26-213dba213ef26e094c16c860ae7f4be0/v3_emdata_clahe_xy/jpeg"
SEG_URL = "gs://manc-seg-v1p2/manc-seg-v1.2"

# A representative 1024³-voxel subvolume near the centre of the reconstructed region.
# Units are 8 nm voxels in (x, y, z) order, matching the CloudVolume coordinate space.
DEFAULT_BOUNDING_BOX = (20000, 21024, 25000, 26024, 40000, 41024)


def _bbox_to_str(bbox):
    return hashlib.md5("_".join(str(v) for v in bbox).encode()).hexdigest()[:12]


def get_manc_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[int, int, int, int, int, int] = DEFAULT_BOUNDING_BOX,
    download: bool = False,
) -> str:
    """Stream a subvolume from the MANC dataset and cache it as an HDF5 file.

    Args:
        path: Filepath to a folder where the cached HDF5 file will be saved.
        bounding_box: The region to fetch as (x_min, x_max, y_min, y_max, z_min, z_max)
            in 8 nm voxel coordinates. Defaults to a central 1024³ training region.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        The filepath to the cached HDF5 file.
    """
    import h5py

    os.makedirs(str(path), exist_ok=True)
    h5_path = os.path.join(str(path), f"{_bbox_to_str(bounding_box)}.h5")
    if os.path.exists(h5_path):
        return h5_path

    if not download:
        raise RuntimeError(
            f"No cached data found at '{h5_path}'. Set download=True to stream it from GCS."
        )

    try:
        import cloudvolume
    except ImportError:
        raise ImportError("The 'cloud-volume' package is required: pip install cloud-volume")

    x_min, x_max, y_min, y_max, z_min, z_max = bounding_box
    print(f"Streaming MANC EM + segmentation for bbox {bounding_box} ...")

    em_vol = cloudvolume.CloudVolume(EM_URL, use_https=True, mip=0, progress=True)
    seg_vol = cloudvolume.CloudVolume(SEG_URL, use_https=True, mip=0, progress=True)

    raw = np.array(em_vol[x_min:x_max, y_min:y_max, z_min:z_max])[..., 0].transpose(2, 1, 0)
    labels = np.array(seg_vol[x_min:x_max, y_min:y_max, z_min:z_max])[..., 0].transpose(2, 1, 0)

    with h5py.File(h5_path, "w", locking=False) as f:
        f.attrs["bounding_box"] = bounding_box
        f.attrs["resolution_nm"] = em_vol.resolution.tolist()
        f.create_dataset("raw", data=raw.astype("uint8"), compression="gzip", chunks=True)
        f.create_dataset("labels", data=labels.astype("uint64"), compression="gzip", chunks=True)

    print(f"Cached to {h5_path} (shape {raw.shape})")
    return h5_path


def get_manc_paths(
    path: Union[os.PathLike, str],
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to MANC HDF5 cache files.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        bounding_boxes: List of regions to fetch, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel coordinates.
            Defaults to [DEFAULT_BOUNDING_BOX].
        download: Whether to stream and cache the data if it is not present.

    Returns:
        List of filepaths to the cached HDF5 files.
    """
    if bounding_boxes is None:
        bounding_boxes = [DEFAULT_BOUNDING_BOX]
    return [get_manc_data(path, bbox, download) for bbox in bounding_boxes]


def get_manc_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the MANC dataset for neuron instance segmentation.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel coordinates.
            Defaults to [DEFAULT_BOUNDING_BOX] - a central 1024³ region.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_manc_paths(path, bounding_boxes, download)

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


def get_manc_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for neuron instance segmentation in the MANC dataset.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel coordinates.
            Defaults to [DEFAULT_BOUNDING_BOX] - a central 1024³ region.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_manc_dataset(
        path, patch_shape, bounding_boxes=bounding_boxes, download=download,
        offsets=offsets, boundaries=boundaries, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
