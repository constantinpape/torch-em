"""The Human Neurons (H01) dataset contains a petascale FIB-SEM volume of human cerebral
cortex with dense automated neuron instance segmentation (C3 release).

The volume covers ~1 mm³ of human temporal cortex at 4 x 4 x 33 nm resolution
(~1.4 PB raw uncompressed). The C3 automated segmentation is provided at 8 x 8 x 33 nm
resolution, covering the same physical region.

The data is hosted on Google Cloud Storage and described in:
Shapson-Coe et al. (2021), https://www.biorxiv.org/content/10.1101/2021.05.29.446289v4.
Please cite this publication if you use the dataset in your research.

NOTE: Accessing this dataset requires the `cloud-volume` package (pip install cloud-volume).

NOTE (on data size): the full volume is 515,892 x 356,400 x 5,293 voxels at 8 x 8 x 33 nm
(~350 TB raw, ~1.4 PB at 4 nm). Downloading the entire volume is not feasible.
Data is instead streamed and cached locally as HDF5 files by specifying bounding boxes
(x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel coordinates.

The volume is highly anisotropic: 8 nm in-plane (xy) and 33 nm in z. Patch shapes should
account for this — e.g. patch_shape=(8, 512, 512) corresponds to a ~264 nm x 4 µm x 4 µm
volume. The full z-extent is only 5,293 slices (~175 µm), so bounding boxes spanning the
complete z range are feasible.
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

# A 2048 × 2048 × 64 subvolume (8 nm xy, 33 nm z) in a neuron-dense region of the cortex.
# Physical size: ~16 µm × 16 µm × 2.1 µm.  Units: 8 nm voxels in (x, y, z) order.
DEFAULT_BOUNDING_BOX = (271360, 273408, 201728, 203776, 2614, 2678)


def _bbox_to_str(bbox):
    """Create a short unique filename stem from a bounding box tuple."""
    key = "_".join(str(v) for v in bbox)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _fetch(cv, x_min, x_max, y_min, y_max, z_min, z_max):
    """Fetch a subvolume and return it as a (z, y, x) array."""
    arr = np.array(cv[x_min:x_max, y_min:y_max, z_min:z_max])[..., 0]
    return arr.transpose(2, 1, 0)


def get_humanneurons_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[int, int, int, int, int, int] = DEFAULT_BOUNDING_BOX,
    download: bool = False,
) -> str:
    """Stream a subvolume from the H01 Human Neurons dataset and cache it as an HDF5 file.

    The HDF5 file contains:
      - raw:    EM grayscale (uint8, 8 nm xy / 33 nm z, z/y/x)
      - labels: neuron instance segmentation (uint64, 8 nm xy / 33 nm z, z/y/x)

    Both layers are stored at the same 8 x 8 x 33 nm resolution. The raw image is
    fetched from the 4 nm source at mip=1 (native 8 nm downsampled scale).

    Args:
        path: Filepath to a folder where the cached HDF5 file will be saved.
        bounding_box: The region to fetch as (x_min, x_max, y_min, y_max, z_min, z_max)
            in 8 nm voxel coordinates. Defaults to a 2048 x 2048 x 64 training region.
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
            "The 'cloud-volume' package is required to access the Human Neurons dataset. "
            "Install it with: 'pip install cloud-volume'."
        )

    x_min, x_max, y_min, y_max, z_min, z_max = bounding_box

    print(f"Streaming H01 Human Neurons EM + segmentation for bbox {bounding_box} ...")

    # EM at mip=1 gives 8×8×33 nm — same resolution as the C3 segmentation at mip=0.
    em_vol = cloudvolume.CloudVolume(EM_URL,  use_https=True, mip=1, progress=True)
    seg_vol = cloudvolume.CloudVolume(SEG_URL, use_https=True, mip=0, progress=True, fill_missing=True)

    raw = _fetch(em_vol,  x_min, x_max, y_min, y_max, z_min, z_max)
    labels = _fetch(seg_vol, x_min, x_max, y_min, y_max, z_min, z_max)

    # Relabel to consecutive integers so IDs fit in uint32 (required for napari and float32 training).
    from skimage.segmentation import relabel_sequential
    labels, _, _ = relabel_sequential(labels)

    resolution_nm = em_vol.mip_resolution(1).tolist()  # [8, 8, 33] nm

    with h5py.File(h5_path, "w") as f:
        f.attrs["bounding_box"] = bounding_box
        f.attrs["crop_size"] = raw.shape  # (z, y, x)
        f.attrs["resolution_nm"] = resolution_nm   # [x, y, z] in nm
        f.create_dataset("raw", data=raw.astype("uint8"),   compression="gzip", chunks=True)
        f.create_dataset("labels", data=labels.astype("uint32"), compression="gzip", chunks=True)

    print(f"Cached to {h5_path}  (raw {raw.shape}, labels {labels.shape})")
    return h5_path


def get_humanneurons_paths(
    path: Union[os.PathLike, str],
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the Human Neurons HDF5 cache files.

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

    return [get_humanneurons_data(path, bbox, download) for bbox in bounding_boxes]


def get_humanneurons_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Human Neurons (H01) dataset for neuron instance segmentation.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
            The volume is anisotropic (8 nm xy, 33 nm z), so small z values are typical,
            e.g. patch_shape=(8, 512, 512).
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel coordinates.
            Defaults to [DEFAULT_BOUNDING_BOX] — a 2048 x 2048 x 64 cortex region.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_humanneurons_paths(path, bounding_boxes, download)

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


def get_humanneurons_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for neuron instance segmentation in the H01 Human Neurons dataset.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
            The volume is anisotropic (8 nm xy, 33 nm z), so small z values are typical,
            e.g. patch_shape=(8, 512, 512).
        batch_size: The batch size for training.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel coordinates.
            Defaults to [DEFAULT_BOUNDING_BOX] — a 2048 x 2048 x 64 cortex region.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_humanneurons_dataset(
        path, patch_shape, bounding_boxes, download, offsets, boundaries, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
