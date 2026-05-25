"""Datasets from the MICrONS (Machine Intelligence from Cortical Networks) project.

Two sources are provided:

**Zenodo training volumes** (basil, minnie, pinky) - https://doi.org/10.5281/zenodo.5760218
  Three EM volumes with sparse neuron instance segmentation and (pinky only) sparse mitochondria
  labels. Most patches contain no annotations. Downloaded as tar.gz archives and cached as HDF5 files.

**minnie65 cubic millimeter** - https://doi.org/10.1038/s41586-025-08790-w
  Full ~1.75 x 1.29 x 1.11 mm volume of mouse primary visual cortex with ~200,000 annotated
  cells. EM and neuron segmentation (version m1300, Jan 2025) are both at 8x8x40 nm native
  (mip=0). Data is streamed from public cloud storage using cloud-volume and cached locally as
  zarr v3 stores (512x4096x4096 vox per box) with sharding and zstd compression.

Please cite the relevant publication if you use either dataset in your research.
"""

import glob
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

import torch_em
from torch.utils.data import DataLoader, Dataset

from .. import util


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

# Mitochondria labels are only present in the pinky volume.
ZENODO_MITO_VOLUMES = ["pinky"]

MINNIE65_EM_URL = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em"
MINNIE65_SEG_URL = "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300/"

# Pre-defined bounding boxes in nm (x_min, x_max, y_min, y_max, z_min, z_max).
# Derived from the same cortical regions used in microns_nuclei (same center coordinates).
# Block size 32768 x 32768 x 20480 nm = 4096x4096x512 vox at 8x8x40 nm (mip=0).
MINNIE65_BOUNDING_BOXES = {
    "train": [
        (384792, 417560, 549540, 582308, 833880, 854360),
        (263368, 296136, 489060, 521828, 836200, 856680),
        (268376, 301144, 562448, 595216, 829560, 850040),
        (279428, 312196, 802124, 834892, 796920, 817400),
        (365248, 398016, 1005708, 1038476, 796920, 817400),
        (462808, 495576, 1054232, 1087000, 796920, 817400),
        (506668, 539436, 1006572, 1039340, 781720, 802200),
        (588344, 621112, 973072, 1005840, 781720, 802200),
    ],
    "val": [
        (733048, 765816, 525324, 558092, 787320, 807800),
        (1131936, 1164704, 618204, 650972, 723880, 744360),
    ],
    "test": [
        (822668, 855436, 414812, 447580, 787320, 807800),
        (986364, 1019132, 398236, 431004, 789320, 809800),
        (1101648, 1134416, 563036, 595804, 789320, 809800),
        (1152312, 1185080, 453124, 485892, 784280, 804760),
    ],
}

MINNIE65_SHARD_SHAPE = (128, 512, 512)
MINNIE65_CHUNK_SHAPE = (64, 128, 128)


def get_microns_data(path: Union[os.PathLike, str], volume: str, download: bool) -> str:
    """Download and extract a single MICrONS Zenodo volume.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        volume: The volume to download. One of 'basil', 'minnie', 'pinky'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the directory containing the extracted HDF5 files.
    """
    assert volume in ZENODO_URLS, f"Invalid volume '{volume}'. Choose from {list(ZENODO_URLS.keys())}."
    os.makedirs(path, exist_ok=True)
    volume_dir = os.path.join(path, volume)
    if not os.path.exists(volume_dir):
        tar_path = os.path.join(path, f"{volume}.tar.gz")
        util.download_source(tar_path, ZENODO_URLS[volume], download, ZENODO_CHECKSUMS[volume])
        util.unzip_tarfile(tar_path, path, remove=True)
    return volume_dir


def get_microns_paths(
    path: Union[os.PathLike, str],
    volumes: Optional[Sequence[str]],
    download: bool,
    label_key: str = "volumes/segmentation",
) -> List[str]:
    """Get paths to MICrONS Zenodo volume HDF5 files.

    Each volume's tar.gz extracts to a subdirectory containing multiple per-volume HDF5 files.
    Files where the image and label shapes do not match are skipped with a warning.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        volumes: The volumes to use. One or more of 'basil', 'minnie', 'pinky'.
            Pass None to use all three volumes.
        download: Whether to download the data if it is not present.
        label_key: HDF5 key for the label array, used to validate shape consistency.

    Returns:
        The filepaths to the stored HDF5 files.
    """
    import h5py

    if volumes is None:
        volumes = list(ZENODO_URLS.keys())
    paths = []
    for vol in volumes:
        vol_dir = get_microns_data(path, vol, download)
        for fpath in sorted(glob.glob(os.path.join(vol_dir, "*.h5"))):
            with h5py.File(fpath, "r") as f:
                if label_key not in f:
                    continue
                img_shape = f["volumes/image"].shape
                lbl_shape = f[label_key].shape
            if img_shape == lbl_shape:
                paths.append(fpath)
            else:
                print(
                    f"Skipping {os.path.basename(fpath)}: image {img_shape} != {label_key} {lbl_shape}"
                )
    return paths


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

    Note: annotations are sparse - most patches contain no labels. Mitochondria labels
    are only available in the pinky volume and are also sparsely annotated.

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
    if label_choice == "mitochondria":
        volumes = [v for v in (volumes or list(ZENODO_URLS.keys())) if v in ZENODO_MITO_VOLUMES]
        if not volumes:
            raise ValueError(f"Mitochondria labels are only available in: {ZENODO_MITO_VOLUMES}.")
    label_key = ZENODO_LABEL_KEYS[label_choice]
    h5_paths = get_microns_paths(path, volumes, download, label_key=label_key)

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


def _minnie65_bbox_to_str(bbox: tuple) -> str:
    return hashlib.md5("_".join(str(v) for v in bbox).encode()).hexdigest()[:12]


def _minnie65_create_array(root, name: str, shape: tuple, dtype, is_label: bool):
    from zarr.codecs import BloscCodec
    shuffle = "bitshuffle" if (np.issubdtype(dtype, np.integer) and is_label) else "shuffle"
    return root.create_array(
        name,
        shape=shape,
        chunks=MINNIE65_CHUNK_SHAPE,
        shards=MINNIE65_SHARD_SHAPE,
        dtype=dtype,
        compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
    )


def _minnie65_bbox_voxels(cv, x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm):
    """Return (x0, x1, y0, y1, z0, z1) voxel bounds and (nz, ny, nx) shape for a CloudVolume."""
    scale = np.array(cv.resolution)
    x0 = int(np.floor(x_min_nm / scale[0]))
    x1 = int(np.ceil(x_max_nm / scale[0]))
    y0 = int(np.floor(y_min_nm / scale[1]))
    y1 = int(np.ceil(y_max_nm / scale[1]))
    z0 = int(np.floor(z_min_nm / scale[2]))
    z1 = int(np.ceil(z_max_nm / scale[2]))
    return x0, x1, y0, y1, z0, z1, (z1 - z0, y1 - y0, x1 - x0)


def _minnie65_download_to_zarr(cv, ds, x0g, y0g, z0g, name: str) -> None:
    """Download a bbox shard-by-shard into a zarr array using a thread pool."""
    shape = ds.shape  # (z, y, x)
    sz, sy, sx = MINNIE65_SHARD_SHAPE

    tasks = []
    for z0_ in range(0, shape[0], sz):
        for y0_ in range(0, shape[1], sy):
            for x0_ in range(0, shape[2], sx):
                z1_ = min(z0_ + sz, shape[0])
                y1_ = min(y0_ + sy, shape[1])
                x1_ = min(x0_ + sx, shape[2])
                tasks.append((
                    (z0_, z1_), (y0_, y1_), (x0_, x1_),
                    (x0g + x0_, x0g + x1_, y0g + y0_, y0g + y1_, z0g + z0_, z0g + z1_),
                ))

    max_workers = 8  # network-bound; more workers with large shards causes OOM

    target_dtype = np.dtype(ds.dtype)

    def worker(item):
        (z0_, z1_), (y0_, y1_), (x0_, x1_), (gx0, gx1, gy0, gy1, gz0, gz1) = item
        block = np.asarray(cv[gx0:gx1, gy0:gy1, gz0:gz1])
        if block.ndim == 4:
            block = block[..., 0]
        ds[z0_:z1_, y0_:y1_, x0_:x1_] = block.transpose(2, 1, 0).astype(target_dtype)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading '{name}'", smoothing=0.05):
            fut.result()


def get_microns_minnie65_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[float, ...],
    em_mip: int = 0,
    seg_mip: int = 0,
    download: bool = False,
) -> str:
    """Stream and cache one minnie65 bounding box as a zarr v3 store.

    The zarr store contains:
      - raw: EM grayscale (uint8, z/y/x)
      - labels: neuron instance segmentation (uint32, z/y/x)

    Both arrays use sharding (shard shape MINNIE65_SHARD_SHAPE, inner chunk shape
    MINNIE65_CHUNK_SHAPE) with zstd+blosc compression. Download is parallelised
    over shards using a thread pool.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        bounding_box: Region in nm as (x_min, x_max, y_min, y_max, z_min, z_max).
        em_mip: MIP level for the EM volume. Default mip=0 gives 8x8x40 nm native resolution.
        seg_mip: MIP level for the segmentation. Default mip=0 gives 8x8x40 nm native resolution.
        download: Whether to stream and cache the data if not present.

    Returns:
        Filepath to the cached zarr store.
    """
    import zarr

    os.makedirs(path, exist_ok=True)
    stem = _minnie65_bbox_to_str(bounding_box)
    zarr_path = os.path.join(str(path), f"{stem}.zarr")

    def _complete(zp):
        return (
            os.path.isdir(os.path.join(zp, "raw"))
            and os.path.isdir(os.path.join(zp, "labels"))
        )

    if _complete(zarr_path):
        return zarr_path
    if not download:
        raise RuntimeError(
            f"No cached data at '{zarr_path}'. Set download=True to stream it from cloud storage."
        )

    try:
        from cloudvolume import CloudVolume
    except ImportError:
        raise ImportError(
            "The 'cloud-volume' package is required to access the minnie65 dataset. "
            "Install it with: pip install cloud-volume"
        )

    x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm = bounding_box
    print(f"Streaming minnie65 bbox {bounding_box} at em_mip={em_mip}, seg_mip={seg_mip} ...")

    em_cv = CloudVolume(MINNIE65_EM_URL, use_https=True, mip=em_mip, progress=False, fill_missing=True)
    seg_cv = CloudVolume(MINNIE65_SEG_URL, use_https=True, mip=seg_mip, progress=False, fill_missing=True)

    ex0, ex1, ey0, ey1, ez0, ez1, em_shape = _minnie65_bbox_voxels(
        em_cv, x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm
    )
    sx0, sx1, sy0, sy1, sz0, sz1, seg_shape = _minnie65_bbox_voxels(
        seg_cv, x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm
    )

    # Use the minimum shape along each axis to handle ceiling-rounding differences.
    shape = tuple(min(e, s) for e, s in zip(em_shape, seg_shape))

    root = zarr.open_group(zarr_path, mode="a")
    root.attrs["bounding_box_nm"] = list(bounding_box)
    root.attrs["em_mip"] = em_mip
    root.attrs["seg_mip"] = seg_mip

    if "raw" not in root:
        ds_raw = _minnie65_create_array(root, "raw", shape, np.dtype("uint8"), is_label=False)
        _minnie65_download_to_zarr(em_cv, ds_raw, ex0, ey0, ez0, name="raw")

    if "labels" not in root:
        ds_lbl = _minnie65_create_array(root, "labels", shape, np.dtype("uint32"), is_label=True)
        _minnie65_download_to_zarr(seg_cv, ds_lbl, sx0, sy0, sz0, name="labels")

    print(f"Cached to {zarr_path} (shape {shape})")
    return zarr_path


def get_microns_minnie65_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "val", "test"]] = None,
    bounding_boxes: Optional[Sequence[Tuple[float, ...]]] = None,
    em_mip: int = 0,
    seg_mip: int = 0,
    download: bool = False,
) -> List[str]:
    """Get paths to cached minnie65 zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        split: Which pre-defined split to use - 'train', 'val', or 'test'.
            Ignored if bounding_boxes is provided. Pass None with no bounding_boxes to use all boxes.
        bounding_boxes: Custom bounding boxes in nm (x_min, x_max, y_min, y_max, z_min, z_max).
            Overrides split-based selection when provided.
        em_mip: MIP level for the EM volume.
        seg_mip: MIP level for the segmentation.
        download: Whether to stream and cache the data if not present.

    Returns:
        Filepaths to the cached zarr stores.
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
    em_mip: int = 0,
    seg_mip: int = 0,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the minnie65 dataset for neuron instance segmentation in EM.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        split: Which pre-defined split to use - 'train', 'val', or 'test'.
            Ignored if bounding_boxes is provided.
        bounding_boxes: Custom bounding boxes in nm (x_min, x_max, y_min, y_max, z_min, z_max).
            Overrides split-based selection when provided.
        em_mip: MIP level for the EM volume. Default mip=0 gives 8x8x40 nm native resolution.
        seg_mip: MIP level for the segmentation. Default mip=0 gives 8x8x40 nm native resolution.
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
    em_mip: int = 0,
    seg_mip: int = 0,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for neuron instance segmentation in the minnie65 dataset.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape (z, y, x) to use for training.
        split: Which pre-defined split to use - 'train', 'val', or 'test'.
            Ignored if bounding_boxes is provided.
        bounding_boxes: Custom bounding boxes in nm (x_min, x_max, y_min, y_max, z_min, z_max).
            Overrides split-based selection when provided.
        em_mip: MIP level for the EM volume. Default mip=0 gives 8x8x40 nm native resolution.
        seg_mip: MIP level for the segmentation. Default mip=0 gives 8x8x40 nm native resolution.
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
