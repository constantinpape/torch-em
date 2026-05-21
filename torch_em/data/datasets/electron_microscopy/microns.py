"""Datasets from the MICrONS (Machine Intelligence from Cortical Networks) project.

Two sources are provided:

**Zenodo training volumes** (basil, minnie, pinky) - https://doi.org/10.5281/zenodo.5760218
  Three dense EM volumes with neuron instance segmentation and optional mitochondria labels,
  downloaded as tar.gz archives and cached as HDF5 files.

**minnie65 cubic millimeter** - https://doi.org/10.1038/s41586-025-08790-w
  Full ~1.75 x 1.29 x 1.11 mm volume of mouse primary visual cortex with ~200,000 annotated
  cells. EM and neuron segmentation (version m1300, Jan 2025) are both at 8x8x40 nm native
  (mip=0); mip=2 gives 32x32x40 nm and is the recommended training resolution. Data is streamed
  from public cloud storage using cloud-volume and cached locally as HDF5 files.

Please cite the relevant publication if you use either dataset in your research.
"""

import hashlib
import os
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

import torch_em
from torch.utils.data import DataLoader, Dataset

from .. import util


# Zenodo training volumes
ZENODO_URLS = {
    "basil": "https://zenodo.org/records/5760218/files/basil.tar.gz?download=1",
    "minnie": "https://zenodo.org/records/5760218/files/minnie.tar.gz?download=1",
    "pinky": "https://zenodo.org/records/5760218/files/pinky.tar.gz?download=1",
}

# SHA256 checksums are not yet available; download will warn but will not fail.
ZENODO_CHECKSUMS = {
    "basil": None,
    "minnie": None,
    "pinky": None,
}

ZENODO_LABEL_KEYS = {
    "neuron": "volumes/segmentation",
    "mitochondria": "volumes/mitochondria",
}

# minnie65 cloud sources
MINNIE65_EM_URL = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em"
MINNIE65_SEG_URL = "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300/"

# Pre-defined bounding boxes in nm (x_min, x_max, y_min, y_max, z_min, z_max).
# Derived from the same cortical regions used in microns_nuclei (same center coordinates).
# Block size 8192 x 8192 x 2560 nm = 256x256x64 vox at 32x32x40 nm (mip=2 for both EM and seg).
MINNIE65_BOUNDING_BOXES = {
    "train": [
        (397080, 405272, 561828, 570020, 842840, 845400),
        (275656, 283848, 501348, 509540, 845160, 847720),
        (280664, 288856, 574736, 582928, 838520, 841080),
        (291716, 299908, 814412, 822604, 805880, 808440),
        (377536, 385728, 1017996, 1026188, 805880, 808440),
        (475096, 483288, 1066520, 1074712, 805880, 808440),
        (518956, 527148, 1018860, 1027052, 790680, 793240),
        (600632, 608824, 985360, 993552, 790680, 793240),
    ],
    "val": [
        (745336, 753528, 537612, 545804, 796280, 798840),
        (1144224, 1152416, 630492, 638684, 732840, 735400),
    ],
    "test": [
        (834956, 843148, 427100, 435292, 796280, 798840),
        (998652, 1006844, 410524, 418716, 798280, 800840),
        (1113936, 1122128, 575324, 583516, 798280, 800840),
        (1164600, 1172792, 465412, 473604, 793240, 795800),
    ],
}


# Zenodo training volumes

def get_microns_data(path: Union[os.PathLike, str], volume: str, download: bool) -> str:
    """Download and extract a single MICrONS Zenodo volume.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        volume: The volume to download. One of 'basil', 'minnie', 'pinky'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the HDF5 file.
    """
    assert volume in ZENODO_URLS, f"Invalid volume '{volume}'. Choose from {list(ZENODO_URLS.keys())}."
    os.makedirs(path, exist_ok=True)
    h5_path = os.path.join(path, f"{volume}.h5")
    if not os.path.exists(h5_path):
        tar_path = os.path.join(path, f"{volume}.tar.gz")
        util.download_source(tar_path, ZENODO_URLS[volume], download, ZENODO_CHECKSUMS[volume])
        util.unzip_tarfile(tar_path, path, remove=True)
    return h5_path


def get_microns_paths(
    path: Union[os.PathLike, str],
    volumes: Optional[Sequence[str]],
    download: bool,
) -> List[str]:
    """Get paths to MICrONS Zenodo volume files.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        volumes: The volumes to use. One or more of 'basil', 'minnie', 'pinky'.
            Pass None to use all three volumes.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the stored HDF5 files.
    """
    if volumes is None:
        volumes = list(ZENODO_URLS.keys())
    return [get_microns_data(path, vol, download) for vol in volumes]


def get_microns_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    volumes: Optional[Sequence[str]] = None,
    label_choice: str = "neuron",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the MICrONS Zenodo dataset for the segmentation of neurons or mitochondria in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        volumes: The volumes to use. One or more of 'basil', 'minnie', 'pinky'.
            Pass None to use all three volumes.
        label_choice: Which labels to segment. One of 'neuron' or 'mitochondria'.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3
    assert label_choice in ZENODO_LABEL_KEYS, \
        f"Invalid label_choice '{label_choice}'. Choose from {list(ZENODO_LABEL_KEYS.keys())}."
    h5_paths = get_microns_paths(path, volumes, download)
    label_key = ZENODO_LABEL_KEYS[label_choice]

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=h5_paths,
        raw_key="volumes/image",
        label_paths=h5_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_microns_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    volumes: Optional[Sequence[str]] = None,
    label_choice: str = "neuron",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for EM neuron or mitochondria segmentation for the MICrONS Zenodo dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        volumes: The volumes to use. One or more of 'basil', 'minnie', 'pinky'.
            Pass None to use all three volumes.
        label_choice: Which labels to segment. One of 'neuron' or 'mitochondria'.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_microns_dataset(
        path=path,
        patch_shape=patch_shape,
        volumes=volumes,
        label_choice=label_choice,
        download=download,
        offsets=offsets,
        boundaries=boundaries,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)


# minnie65 cubic millimeter

def _minnie65_bbox_to_str(bbox: tuple) -> str:
    return hashlib.md5("_".join(str(v) for v in bbox).encode()).hexdigest()[:12]


def _minnie65_fetch_zyx(cv, x_min_nm: float, x_max_nm: float,
                        y_min_nm: float, y_max_nm: float,
                        z_min_nm: float, z_max_nm: float) -> np.ndarray:
    """Download a nm-defined sub-region and return it as a (z, y, x) array."""
    scale = np.array(cv.resolution)  # [x, y, z] nm at the initialized mip
    x0 = int(np.floor(x_min_nm / scale[0]))
    x1 = int(np.ceil(x_max_nm / scale[0]))
    y0 = int(np.floor(y_min_nm / scale[1]))
    y1 = int(np.ceil(y_max_nm / scale[1]))
    z0 = int(np.floor(z_min_nm / scale[2]))
    z1 = int(np.ceil(z_max_nm / scale[2]))
    arr = np.array(cv[x0:x1, y0:y1, z0:z1])
    if arr.ndim == 4:
        arr = arr[..., 0]
    return arr.transpose(2, 1, 0).copy()  # (x,y,z) -> (z,y,x)


def get_microns_minnie65_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[float, ...],
    em_mip: int = 2,
    seg_mip: int = 2,
    download: bool = False,
) -> str:
    """Stream and cache one minnie65 bounding box as an HDF5 file.

    The HDF5 file contains:
      - raw: EM grayscale (uint8, z/y/x)
      - labels: neuron instance segmentation (uint32, z/y/x)

    Both arrays are at the physical resolution defined by em_mip/seg_mip. The defaults
    em_mip=2 and seg_mip=2 both give 32x32x40 nm (native resolution is 8x8x40 nm at mip=0).
    Adjust if using a different mip combination.

    Args:
        path: Filepath to a folder where the cached HDF5 file will be saved.
        bounding_box: Region in nm as (x_min, x_max, y_min, y_max, z_min, z_max).
        em_mip: MIP level for the EM volume. mip=2 gives 32x32x40 nm (native is 8nm at mip=0).
        seg_mip: MIP level for the segmentation. mip=2 gives 32x32x40 nm (native is 8nm at mip=0).
        download: Whether to stream and cache the data if not present.

    Returns:
        Filepath to the cached HDF5 file.
    """
    import h5py

    os.makedirs(path, exist_ok=True)
    stem = _minnie65_bbox_to_str(bounding_box)
    h5_path = os.path.join(str(path), f"{stem}.h5")
    if os.path.exists(h5_path):
        return h5_path
    if not download:
        raise RuntimeError(
            f"No cached data at '{h5_path}'. Set download=True to stream it from cloud storage."
        )

    try:
        from cloudvolume import CloudVolume
    except ImportError:
        raise ImportError(
            "The 'cloud-volume' package is required to access the minnie65 dataset. "
            "Install it with: pip install cloud-volume"
        )

    x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm = bounding_box
    print(f"Streaming minnie65 EM + segmentation for bbox {bounding_box} ...")

    em_cv = CloudVolume(MINNIE65_EM_URL, use_https=True, mip=em_mip, progress=True, fill_missing=True)
    seg_cv = CloudVolume(MINNIE65_SEG_URL, use_https=True, mip=seg_mip, progress=True, fill_missing=True)

    raw = _minnie65_fetch_zyx(em_cv, x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm)
    labels = _minnie65_fetch_zyx(seg_cv, x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm)

    # Trim to matching shape if rounding gives a 1-voxel difference.
    min_shape = tuple(min(r, l) for r, l in zip(raw.shape, labels.shape))
    if raw.shape != labels.shape:
        print(f"Shapes differ ({raw.shape} vs {labels.shape}); trimming to {min_shape}.")
        raw = raw[:min_shape[0], :min_shape[1], :min_shape[2]]
        labels = labels[:min_shape[0], :min_shape[1], :min_shape[2]]

    if raw.shape != labels.shape:
        raise RuntimeError(
            f"EM shape {raw.shape} still doesn't match segmentation shape {labels.shape} after trim. "
            "Check that em_mip and seg_mip correspond to the same physical resolution."
        )

    from skimage.segmentation import relabel_sequential
    labels, _, _ = relabel_sequential(labels)

    with h5py.File(h5_path, "w", locking=False) as f:
        f.attrs["bounding_box_nm"] = bounding_box
        f.attrs["em_mip"] = em_mip
        f.attrs["seg_mip"] = seg_mip
        f.create_dataset("raw", data=raw.astype("uint8"), compression="gzip", chunks=True)
        f.create_dataset("labels", data=labels.astype("uint32"), compression="gzip", chunks=True)

    print(f"Cached to {h5_path} (shape {raw.shape})")
    return h5_path


def get_microns_minnie65_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "val", "test"]] = None,
    bounding_boxes: Optional[Sequence[Tuple[float, ...]]] = None,
    em_mip: int = 2,
    seg_mip: int = 2,
    download: bool = False,
) -> List[str]:
    """Get paths to cached minnie65 HDF5 files.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        split: Which pre-defined split to use - 'train', 'val', or 'test'.
            Ignored if bounding_boxes is provided. Pass None with no bounding_boxes to use all boxes.
        bounding_boxes: Custom bounding boxes in nm (x_min, x_max, y_min, y_max, z_min, z_max).
            Overrides split-based selection when provided.
        em_mip: MIP level for the EM volume.
        seg_mip: MIP level for the segmentation.
        download: Whether to stream and cache the data if not present.

    Returns:
        Filepaths to the cached HDF5 files.
    """
    if bounding_boxes is not None:
        boxes = list(bounding_boxes)
    elif split is not None:
        assert split in MINNIE65_BOUNDING_BOXES, \
            f"Invalid split '{split}'. Choose from {list(MINNIE65_BOUNDING_BOXES.keys())}."
        boxes = MINNIE65_BOUNDING_BOXES[split]
    else:
        boxes = [bb for split_boxes in MINNIE65_BOUNDING_BOXES.values() for bb in split_boxes]
    return [get_microns_minnie65_data(path, bb, em_mip, seg_mip, download) for bb in boxes]


def get_microns_minnie65_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    split: Optional[Literal["train", "val", "test"]] = None,
    bounding_boxes: Optional[Sequence[Tuple[float, ...]]] = None,
    em_mip: int = 2,
    seg_mip: int = 2,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the minnie65 dataset for neuron instance segmentation in EM.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        split: Which pre-defined split to use - 'train', 'val', or 'test'.
            Ignored if bounding_boxes is provided.
        bounding_boxes: Custom bounding boxes in nm (x_min, x_max, y_min, y_max, z_min, z_max).
            Overrides split-based selection when provided.
        em_mip: MIP level for the EM volume. mip=2 gives 32x32x40 nm (native is 8nm at mip=0).
        seg_mip: MIP level for the segmentation. mip=2 gives 32x32x40 nm (native is 8nm at mip=0).
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_microns_minnie65_paths(path, split, bounding_boxes, em_mip, seg_mip, download)

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


def get_microns_minnie65_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    split: Optional[Literal["train", "val", "test"]] = None,
    bounding_boxes: Optional[Sequence[Tuple[float, ...]]] = None,
    em_mip: int = 2,
    seg_mip: int = 2,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for neuron instance segmentation in the minnie65 dataset.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape (z, y, x) to use for training.
        split: Which pre-defined split to use - 'train', 'val', or 'test'.
            Ignored if bounding_boxes is provided.
        bounding_boxes: Custom bounding boxes in nm (x_min, x_max, y_min, y_max, z_min, z_max).
            Overrides split-based selection when provided.
        em_mip: MIP level for the EM volume. mip=2 gives 32x32x40 nm (native is 8nm at mip=0).
        seg_mip: MIP level for the segmentation. mip=2 gives 32x32x40 nm (native is 8nm at mip=0).
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_microns_minnie65_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        bounding_boxes=bounding_boxes,
        em_mip=em_mip,
        seg_mip=seg_mip,
        download=download,
        offsets=offsets,
        boundaries=boundaries,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
