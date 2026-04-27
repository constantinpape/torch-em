"""The Male CNS Connectome dataset contains a full FIB-SEM volume of the Drosophila
male central nervous system with dense neuron instance segmentation.

It covers the central brain, optic lobes, and ventral nerve cord at 8nm isotropic
resolution, with ~135k neurons reconstructed and proofread.

The data is hosted at https://male-cns.janelia.org and accessible via Google Cloud Storage.
The EM volume is at gs://flyem-male-cns/em/em-clahe-jpeg and the segmentation is at
gs://flyem-male-cns/v0.9/segmentation.

The dataset is described at https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2.
Please cite this publication if you use the dataset in your research.

NOTE: accessing this dataset requires the `cloud-volume` package (pip install cloud-volume).

NOTE (on data size): the full volume is (94088, 78317, 134576) voxels at 8nm isotropic
resolution (~978 TB raw uncompressed, ~8 PB labels uncompressed). Downloading the entire
volume is not feasible. Data is instead accessed by specifying bounding boxes
(x_min, x_max, y_min, y_max, z_min, z_max) in 8nm voxel coordinates, which are streamed
from GCS and cached locally as HDF5 files (~1.2 GB per 1024³ subvolume). To target
specific regions of the CNS (central brain, optic lobes, VNC), the neuropil ROI volume
at gs://flyem-male-cns/rois/fullbrain-roi-v4 can be used to identify relevant coordinate
ranges.
"""

import os
import hashlib
from typing import List, Optional, Tuple, Union

import numpy as np

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


EM_URL = "gs://flyem-male-cns/em/em-clahe-jpeg"
SEG_URL = "gs://flyem-male-cns/v0.9/segmentation"

# A representative 1024³-voxel subvolume near the centre of the well-reconstructed region.
# Units are 8 nm voxels in (x, y, z) order, matching the CloudVolume coordinate space.
DEFAULT_BOUNDING_BOX = (40000, 41024, 40000, 41024, 20000, 21024)


def _bbox_to_str(bbox):
    """Create a short unique filename stem from a bounding box tuple."""
    key = "_".join(str(v) for v in bbox)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def get_malecns_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[int, int, int, int, int, int] = DEFAULT_BOUNDING_BOX,
    download: bool = False,
) -> str:
    """Stream a subvolume from the Male CNS Connectome and cache it as an HDF5 file.

    Args:
        path: Filepath to a folder where the cached HDF5 file will be saved.
        bounding_box: The region to fetch as (x_min, x_max, y_min, y_max, z_min, z_max)
            in 8 nm voxel coordinates. Defaults to a central 1024³ training region.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        The filepath to the cached HDF5 file.
    """
    import h5py

    os.makedirs(path, exist_ok=True)

    stem = _bbox_to_str(bounding_box)
    h5_path = os.path.join(path, f"{stem}.h5")
    if os.path.exists(h5_path):
        return h5_path

    if not download:
        raise RuntimeError(
            f"No cached data found at '{h5_path}'. Set download=True to stream it from GCS."
        )

    try:
        import cloudvolume
    except ImportError:
        raise ImportError(
            "The 'cloud-volume' package is required to access the Male CNS dataset. "
            "Install it with: 'pip install cloud-volume'."
        )

    x_min, x_max, y_min, y_max, z_min, z_max = bounding_box

    print(f"Streaming Male CNS EM + segmentation for bbox {bounding_box} ...")

    em_vol = cloudvolume.CloudVolume(EM_URL, use_https=True, mip=0, progress=True)
    seg_vol = cloudvolume.CloudVolume(SEG_URL, use_https=True, mip=0, progress=True)

    # cloud-volume returns (x, y, z, 1). Let's squeeze channel and transpose to (z, y, x).
    raw = np.array(em_vol[x_min:x_max, y_min:y_max, z_min:z_max])[..., 0].transpose(2, 1, 0)
    labels = np.array(seg_vol[x_min:x_max, y_min:y_max, z_min:z_max])[..., 0].transpose(2, 1, 0)

    resolution_nm = em_vol.resolution.tolist()  # [x, y, z] in nm

    with h5py.File(h5_path, "w", locking=False) as f:
        f.attrs["bounding_box"] = bounding_box
        f.attrs["crop_size"] = raw.shape  # (z, y, x) after transpose
        f.attrs["resolution_nm"] = resolution_nm  # [x, y, z] in nm
        f.create_dataset("raw", data=raw.astype("uint8"), compression="gzip", chunks=True)
        f.create_dataset("labels", data=labels.astype("uint64"), compression="gzip", chunks=True)

    print(f"Cached to {h5_path}  (raw {raw.shape}, labels {labels.shape})")
    return h5_path


def get_malecns_paths(
    path: Union[os.PathLike, str],
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the Male CNS HDF5 cache files.

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

    return [get_malecns_data(path, bbox, download) for bbox in bounding_boxes]


def get_malecns_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Male CNS Connectome dataset for neuron instance segmentation.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel coordinates.
            Defaults to [DEFAULT_BOUNDING_BOX] — a central 1024³ region.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_malecns_paths(path, bounding_boxes, download)

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


def get_malecns_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for neuron instance segmentation in the Male CNS Connectome.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel coordinates.
            Defaults to [DEFAULT_BOUNDING_BOX] — a central 1024³ region.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_malecns_dataset(
        path, patch_shape, bounding_boxes=bounding_boxes, download=download,
        offsets=offsets, boundaries=boundaries, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
