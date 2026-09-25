"""The LICONN dataset contains a dense connectomic reconstruction of mouse hippocampal
CA1 neuropil acquired by spinning-disk confocal microscopy of expansion-microscopy-processed
tissue (~16x expansion), yielding a native voxel resolution of 9x9x12 nm (XYZ) at mip=0.
All neuronal structures are densely annotated as instance segmentations: 18,268 axons
(342 mm total length), 1,643 dendrites (119 mm total), and 71,269 spines.

Two segmentation variants are provided:
- 'proofread': manually proofread segmentation (higher accuracy).
- 'agglomerated': automatically agglomerated segmentation.

The data is served as Neuroglancer precomputed volumes from Google Cloud Storage
(gs://liconn-public) and requires the cloudvolume package to download.

All volumes are stored in a single zarr v3 store (liconn.zarr) with sharding.
The store contains arrays 'raw', 'seg_proofread', and 'seg_agglomerated'.

This dataset is from the following publication:
- Velicky et al. (2025): https://doi.org/10.1038/s41586-025-08985-1
Please cite it if you use this dataset in your research.
"""

import os
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

SEGMENTATIONS = ("proofread", "agglomerated")

ZARR_FNAME = "liconn.zarr"
SHARD_SHAPE = (64, 256, 256)
CHUNK_SHAPE = (32, 64, 64)


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
    return root.create_array(
        name,
        shape=shape,
        chunks=CHUNK_SHAPE,
        shards=SHARD_SHAPE,
        dtype=dtype,
        compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
    )


def _download_ng_volume(vol, ds, name: str) -> None:
    x0, y0, z0 = map(int, vol.bounds.minpt)
    x1, y1, z1 = map(int, vol.bounds.maxpt)
    shape = (z1 - z0, y1 - y0, x1 - x0)

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

    max_workers = max(8, (os.cpu_count() or 4) * 4)

    def worker(item):
        (z0_, z1_), (y0_, y1_), (x0_, x1_), (gx0, gx1, gy0, gy1, gz0, gz1) = item
        block = np.asarray(vol[gx0:gx1, gy0:gy1, gz0:gz1])
        ds[z0_:z1_, y0_:y1_, x0_:x1_] = _to_zyx(block)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading '{name}'", smoothing=0.05):
            fut.result()


def get_liconn_data(
    path: Union[os.PathLike, str],
    segmentation: str = "proofread",
    download: bool = False,
) -> None:
    """Download the LICONN image and segmentation into a single zarr v3 store with sharding.

    The entire volume is always downloaded (image at mip=1, 18x18x24 nm resolution;
    segmentation at mip=0, same voxel grid). ROI-based sub-region selection is not supported
    at download time - use the roi parameter in get_liconn_dataset to restrict patch sampling
    to a sub-region after the full volume is on disk.

    All arrays are stored in liconn.zarr with names 'raw', 'seg_proofread', and
    'seg_agglomerated'. Each array uses zarr v3 sharding (shard shape SHARD_SHAPE,
    inner chunk shape CHUNK_SHAPE) with zstd+blosc compression.

    Args:
        path: Filepath to a folder where the data will be saved.
        segmentation: Which segmentation variant to download. Either 'proofread' or 'agglomerated'.
        download: Whether to download the data if it is not present.
    """
    if segmentation not in SEGMENTATIONS:
        raise ValueError(f"'{segmentation}' is not a valid segmentation. Choose from {SEGMENTATIONS}.")

    try:
        from cloudvolume import CloudVolume
    except ImportError:
        raise ImportError(
            "cloudvolume is required to download the LICONN data. Install it with: pip install cloud-volume"
        )

    import zarr

    os.makedirs(path, exist_ok=True)
    zarr_path = os.path.join(str(path), ZARR_FNAME)
    label_key = f"seg_{segmentation}"

    def _array_complete(arr_name):
        d = os.path.join(zarr_path, arr_name)
        return os.path.isdir(d) and len(os.listdir(d)) > 1

    raw_missing = not _array_complete("raw")
    label_missing = not _array_complete(label_key)

    if not raw_missing and not label_missing:
        return

    if not download:
        missing = [k for k, m in [("raw", raw_missing), (label_key, label_missing)] if m]
        raise RuntimeError(f"LICONN arrays {missing} not found in {zarr_path}. Pass download=True to download them.")

    root = zarr.open_group(zarr_path, mode="a")

    if raw_missing:
        img_cv = CloudVolume(IMG_URL, mip=1, progress=False, cache=False, fill_missing=True)
        x0, y0, z0 = map(int, img_cv.bounds.minpt)
        x1, y1, z1 = map(int, img_cv.bounds.maxpt)
        shape = (z1 - z0, y1 - y0, x1 - x0)
        ds = _create_array(root, "raw", shape, np.dtype(img_cv.dtype), is_label=False)
        _download_ng_volume(img_cv, ds, name="raw")

    if label_missing:
        seg_url = SEG_PR_URL if segmentation == "proofread" else SEG_AGG_URL
        seg_cv = CloudVolume(seg_url, mip=0, progress=False, cache=False, fill_missing=True)
        x0, y0, z0 = map(int, seg_cv.bounds.minpt)
        x1, y1, z1 = map(int, seg_cv.bounds.maxpt)
        shape = (z1 - z0, y1 - y0, x1 - x0)
        ds = _create_array(root, label_key, shape, np.dtype(seg_cv.dtype), is_label=True)
        _download_ng_volume(seg_cv, ds, name=label_key)


def get_liconn_paths(
    path: Union[os.PathLike, str],
    segmentation: str = "proofread",
    download: bool = False,
) -> str:
    """Get the filepath to the LICONN zarr store.

    The store contains arrays 'raw', 'seg_proofread', and 'seg_agglomerated'.

    Args:
        path: Filepath to a folder where the data will be saved.
        segmentation: Which segmentation variant to ensure is present. Either 'proofread' or 'agglomerated'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the liconn.zarr store.
    """
    get_liconn_data(path, segmentation, download)
    return os.path.join(str(path), ZARR_FNAME)


def get_liconn_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    segmentation: str = "proofread",
    roi: Optional[Tuple[slice, ...]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the LICONN dataset for neuron instance segmentation in expansion microscopy.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape to use for training.
        segmentation: Which segmentation variant to use. Either 'proofread' or 'agglomerated'.
        roi: Optional region-of-interest as a tuple of slices (Z, Y, X) restricting which part
            of the already-downloaded volume is used for patch sampling. The full volume is
            always downloaded regardless of this parameter.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    zarr_path = get_liconn_paths(path, segmentation, download)
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
    segmentation: str = "proofread",
    roi: Optional[Tuple[slice, ...]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for the LICONN dataset for neuron instance segmentation.

    Args:
        path: Filepath to a folder where the data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        segmentation: Which segmentation variant to use. Either 'proofread' or 'agglomerated'.
        roi: Optional region-of-interest as a tuple of slices (Z, Y, X) restricting which part
            of the already-downloaded volume is used for patch sampling. The full volume is
            always downloaded regardless of this parameter.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_liconn_dataset(path, patch_shape, segmentation, roi, download, offsets, boundaries, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
