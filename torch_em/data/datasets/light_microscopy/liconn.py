"""The LICONN dataset contains a dense connectomic reconstruction of mouse hippocampal
CA1 neuropil acquired by spinning-disk confocal microscopy of expansion-microscopy-processed
tissue (~16x expansion), yielding a native voxel resolution of 9x9x12 nm (XYZ) at mip=0.
All neuronal structures are densely annotated as instance segmentations: 18,268 axons
(342 mm total length), 1,643 dendrites (119 mm total), and 71,269 spines.

Two volumes are available, selected via the 'volume' argument:
- 'fig2' (default, ExPID82_1): the densely reconstructed CA1 volume from Fig. 2 described above,
  at 18x18x24 nm. Two segmentation variants are provided:
    - 'proofread': manually proofread segmentation (higher accuracy).
    - 'agglomerated': automatically agglomerated segmentation.
  Stored in liconn.zarr with arrays 'raw', 'seg_proofread', and 'seg_agglomerated'.
- 'fig1' (ExPID124): the large hippocampal volume from Fig. 1a, at 20x20x25 nm. Its labeled region
  is 20156x5724x997 voxels (XYZ), ~8x larger than 'fig2'. NOTE: neither of its segmentations
  is proofread dense ground truth:
    - 'sparse': reconstructions of ~80 selected neurons (somata and neurites), covering ~6% of
      the volume. Everything else is 0, so unlabeled neurites are NOT background.
    - 'dense_auto': the dense automatic segmentation, which is unproofread and over-split
      (e.g. nuclei are broken into many fragments).
  Stored in liconn_fig1.zarr with arrays 'raw', 'seg_sparse', and 'seg_dense_auto'.

The data is served as Neuroglancer precomputed volumes from Google Cloud Storage
(gs://liconn-public) and requires the cloudvolume package to download.
For 'fig1' only the bounding box that contains labels is downloaded, which drops the unannotated
margins. 'fig2' is downloaded in full to keep its voxel coordinates stable; its labeled bounding box
is available as FIG2_LABELED_ROI and can be passed as 'roi'. All volumes are stored in zarr v3 stores
with sharding.

This dataset is from the following publication:
- Velicky et al. (2025): https://doi.org/10.1038/s41586-025-08985-1
Please cite it if you use this dataset in your research.
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


IMG_URL = "precomputed://https://storage.googleapis.com/liconn-public/ExPID82_1/image_230130b"
SEG_PR_URL = "precomputed://https://storage.googleapis.com/liconn-public/ExPID82_1/segmentation/231030_agg_240123"
SEG_AGG_URL = "precomputed://https://storage.googleapis.com/liconn-public/ExPID82_1/segmentation/231030_agg_230921_cmpl"

FIG1_URL = "precomputed://https://storage.googleapis.com/liconn-public/ExPID124"
FIG1_IMG_URL = f"{FIG1_URL}/image"
FIG1_SEG_SPARSE_URL = f"{FIG1_URL}/seg_230907_agglo231128_combined2"
FIG1_SEG_DENSE_URL = f"{FIG1_URL}/seg_230907"

ZARR_FNAME = "liconn.zarr"

# The (Z, Y, X) region of 'fig2' that contains labels, computed from the full resolution segmentations.
FIG2_LABELED_ROI = (slice(72, 627), slice(185, 4535), slice(140, 3690))

# 'bbox' is the downloaded (half-open, ZYX) voxel range; 'fig2' stays at full extent to keep its coordinates.
VOLUMES = {
    "fig2": {
        "zarr": ZARR_FNAME,
        "image": IMG_URL,
        "segmentations": {"proofread": SEG_PR_URL, "agglomerated": SEG_AGG_URL},
        "bbox": ((0, 795), (0, 4870), (0, 3825)),
        "accept_unflagged": True,
    },
    "fig1": {
        "zarr": "liconn_fig1.zarr",
        "image": FIG1_IMG_URL,
        "segmentations": {"sparse": FIG1_SEG_SPARSE_URL, "dense_auto": FIG1_SEG_DENSE_URL},
        "bbox": ((0, 997), (204, 5928), (1180, 21336)),
    },
}
SHARD_SHAPE = (64, 256, 256)
CHUNK_SHAPE = (32, 64, 64)

# Marks arrays whose download finished, so that interrupted downloads are detected.
COMPLETE_ATTR = "download_complete"


def _to_zyx(a: np.ndarray) -> np.ndarray:
    # CloudVolume returns (X, Y, Z[, C]); squeeze trailing channel dim if present.
    if a.ndim == 4:
        a = a.squeeze(axis=-1)
    if a.ndim != 3:
        raise ValueError(f"Expected 3D block, got shape {a.shape}")
    return a.transpose(2, 1, 0)


def _create_array(root, name: str, shape, dtype, is_label: bool):
    from zarr.codecs import BloscCodec

    shuffle = "bitshuffle" if (np.issubdtype(dtype, np.integer) and is_label) else "shuffle"
    ds = root.create_array(
        name,
        shape=shape,
        chunks=CHUNK_SHAPE,
        shards=SHARD_SHAPE,
        dtype=dtype,
        compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        overwrite=True,  # Replaces leftovers from an interrupted download.
    )
    ds.attrs[COMPLETE_ATTR] = False
    return ds


def _array_complete(zarr_path: str, name: str, shape: Tuple[int, ...], accept_unflagged: bool = False) -> bool:
    meta_path = os.path.join(zarr_path, name, "zarr.json")
    if not os.path.isfile(meta_path):
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    # A different shape means the array was not cropped to the current bounding box.
    if tuple(meta["shape"]) != tuple(shape):
        return False
    complete = meta.get("attributes", {}).get(COMPLETE_ATTR)
    if complete is None and accept_unflagged:
        # Downloaded before the completion flag existed.
        return len(os.listdir(os.path.dirname(meta_path))) > 1
    return complete is True


def _download_ng_volume(vol, ds, name: str, bbox) -> None:
    # The bounding box is given relative to the start of the volume.
    vx0, vy0, vz0 = map(int, vol.bounds.minpt)
    (bz0, bz1), (by0, by1), (bx0, bx1) = bbox
    x0, y0, z0 = vx0 + bx0, vy0 + by0, vz0 + bz0
    shape = (bz1 - bz0, by1 - by0, bx1 - bx0)

    tasks = []
    for z0_ in range(0, shape[0], SHARD_SHAPE[0]):
        for y0_ in range(0, shape[1], SHARD_SHAPE[1]):
            for x0_ in range(0, shape[2], SHARD_SHAPE[2]):
                z1_ = min(z0_ + SHARD_SHAPE[0], shape[0])
                y1_ = min(y0_ + SHARD_SHAPE[1], shape[1])
                x1_ = min(x0_ + SHARD_SHAPE[2], shape[2])
                tasks.append((
                    (z0_, z1_), (y0_, y1_), (x0_, x1_),
                    (x0 + x0_, x0 + x1_, y0 + y0_, y0 + y1_, z0 + z0_, z0 + z1_)
                ))

    # os.cpu_count() counts all cores of the node; more than ~32 threads only costs memory.
    n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
    max_workers = min(32, 4 * n_cpus)

    def worker(item):
        (z0_, z1_), (y0_, y1_), (x0_, x1_), (gx0, gx1, gy0, gy1, gz0, gz1) = item
        block = np.asarray(vol[gx0:gx1, gy0:gy1, gz0:gz1])
        ds[z0_:z1_, y0_:y1_, x0_:x1_] = _to_zyx(block)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading '{name}'", smoothing=0.05):
            fut.result()

    ds.attrs[COMPLETE_ATTR] = True


def _get_segmentation(volume: str, segmentation: Optional[str]) -> str:
    if volume not in VOLUMES:
        raise ValueError(f"'{volume}' is not a valid volume. Choose from {tuple(VOLUMES)}.")
    choices = tuple(VOLUMES[volume]["segmentations"])
    if segmentation is None:
        return choices[0]
    if segmentation not in choices:
        raise ValueError(f"'{segmentation}' is not a valid segmentation for '{volume}'. Choose from {choices}.")
    return segmentation


def get_liconn_data(
    path: Union[os.PathLike, str],
    segmentation: Optional[str] = None,
    download: bool = False,
    volume: str = "fig2",
) -> None:
    """Download the LICONN image and segmentation into a single zarr v3 store with sharding.

    The full 'fig2' volume and the bounding box containing all labels of 'fig1' are downloaded (image at
    mip=1, i.e. 18x18x24 nm for 'fig2' and 20x20x25 nm for 'fig1'; segmentation at mip=0, same voxel grid).
    Further ROI-based sub-region selection is not supported at download time - use the roi parameter in
    get_liconn_dataset to restrict patch sampling to a sub-region after the data is on disk.

    The 'fig2' arrays are stored in liconn.zarr ('raw', 'seg_proofread', 'seg_agglomerated'), the 'fig1'
    arrays in liconn_fig1.zarr ('raw', 'seg_sparse', 'seg_dense_auto'). Each array uses zarr v3 sharding
    (shard shape SHARD_SHAPE, inner chunk shape CHUNK_SHAPE) with zstd+blosc compression.
    An array only counts as present once its download has finished; an interrupted download
    is restarted from scratch.

    Args:
        path: Filepath to a folder where the data will be saved.
        segmentation: Which segmentation variant to download. Either 'proofread' or 'agglomerated' for 'fig2',
            or 'sparse' or 'dense_auto' for 'fig1'. By default the first of these is used.
        download: Whether to download the data if it is not present.
        volume: Which volume to download. Either 'fig2' or 'fig1'.
    """
    segmentation = _get_segmentation(volume, segmentation)
    config = VOLUMES[volume]

    os.makedirs(path, exist_ok=True)
    zarr_path = os.path.join(str(path), config["zarr"])
    label_key = f"seg_{segmentation}"
    bbox = config["bbox"]
    shape = tuple(stop - start for start, stop in bbox)

    accept_unflagged = config.get("accept_unflagged", False)
    raw_missing = not _array_complete(zarr_path, "raw", shape, accept_unflagged)
    label_missing = not _array_complete(zarr_path, label_key, shape, accept_unflagged)

    if not raw_missing and not label_missing:
        return

    if not download:
        missing = [k for k, m in [("raw", raw_missing), (label_key, label_missing)] if m]
        raise RuntimeError(
            f"LICONN arrays {missing} not found or incomplete in {zarr_path}. Pass download=True to download them."
        )

    # Only the download needs cloudvolume, so data already on disk loads without it.
    try:
        from cloudvolume import CloudVolume
    except ImportError:
        raise ImportError(
            "cloudvolume is required to download the LICONN data. Install it with: pip install cloud-volume"
        )

    import zarr

    root = zarr.open_group(zarr_path, mode="a")

    if raw_missing:
        img_cv = CloudVolume(config["image"], mip=1, progress=False, cache=False, fill_missing=True)
        ds = _create_array(root, "raw", shape, np.dtype(img_cv.dtype), is_label=False)
        _download_ng_volume(img_cv, ds, name="raw", bbox=bbox)

    if label_missing:
        seg_url = config["segmentations"][segmentation]
        seg_cv = CloudVolume(seg_url, mip=0, progress=False, cache=False, fill_missing=True)
        ds = _create_array(root, label_key, shape, np.dtype(seg_cv.dtype), is_label=True)
        _download_ng_volume(seg_cv, ds, name=label_key, bbox=bbox)


def get_liconn_paths(
    path: Union[os.PathLike, str],
    segmentation: Optional[str] = None,
    download: bool = False,
    volume: str = "fig2",
) -> str:
    """Get the filepath to the LICONN zarr store.

    The 'fig2' store (liconn.zarr) contains arrays 'raw', 'seg_proofread', and 'seg_agglomerated',
    the 'fig1' store (liconn_fig1.zarr) contains arrays 'raw', 'seg_sparse', and 'seg_dense_auto'.

    Args:
        path: Filepath to a folder where the data will be saved.
        segmentation: Which segmentation variant to ensure is present. Either 'proofread' or 'agglomerated'
            for 'fig2', or 'sparse' or 'dense_auto' for 'fig1'. By default the first of these is used.
        download: Whether to download the data if it is not present.
        volume: Which volume to use. Either 'fig2' or 'fig1'.

    Returns:
        Filepath to the zarr store.
    """
    get_liconn_data(path, segmentation, download, volume)
    return os.path.join(str(path), VOLUMES[volume]["zarr"])


def get_liconn_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    segmentation: Optional[str] = None,
    roi: Optional[Tuple[slice, ...]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    volume: str = "fig2",
    **kwargs,
) -> Dataset:
    """Get the LICONN dataset for neuron instance segmentation in expansion microscopy.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape to use for training.
        segmentation: Which segmentation variant to use. Either 'proofread' or 'agglomerated' for 'fig2',
            or 'sparse' or 'dense_auto' for 'fig1'. By default the first of these is used.
        roi: Optional region-of-interest as a tuple of slices (Z, Y, X) restricting which part
            of the already-downloaded data is used for patch sampling. The data is always downloaded
            regardless of this parameter. For 'fig2', FIG2_LABELED_ROI restricts sampling to the labeled region.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        volume: Which volume to use. Either 'fig2' (dense proofread CA1 volume) or 'fig1'
            (large volume with sparse or unproofread segmentations, see the module docstring).
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    segmentation = _get_segmentation(volume, segmentation)
    zarr_path = get_liconn_paths(path, segmentation, download, volume)
    label_key = f"seg_{segmentation}"

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets
    )
    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=zarr_path,
        raw_key="raw",
        label_paths=zarr_path,
        label_key=label_key,
        patch_shape=patch_shape,
        rois=roi,
        **kwargs,
    )


def get_liconn_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    segmentation: Optional[str] = None,
    roi: Optional[Tuple[slice, ...]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    volume: str = "fig2",
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for the LICONN dataset for neuron instance segmentation.

    Args:
        path: Filepath to a folder where the data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        segmentation: Which segmentation variant to use. Either 'proofread' or 'agglomerated' for 'fig2',
            or 'sparse' or 'dense_auto' for 'fig1'. By default the first of these is used.
        roi: Optional region-of-interest as a tuple of slices (Z, Y, X) restricting which part
            of the already-downloaded data is used for patch sampling. The data is always downloaded
            regardless of this parameter. For 'fig2', FIG2_LABELED_ROI restricts sampling to the labeled region.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        volume: Which volume to use. Either 'fig2' (dense proofread CA1 volume) or 'fig1'
            (large volume with sparse or unproofread segmentations, see the module docstring).
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_liconn_dataset(
        path, patch_shape, segmentation, roi, download, offsets, boundaries, volume, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
