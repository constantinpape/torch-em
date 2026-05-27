"""Hydra vulgaris endodermal nerve net dataset for neuron instance segmentation in FIB-SEM.

The dataset contains a single FIB-SEM volume of the endodermal nerve net of Hydra vulgaris,
with 20 completely reconstructed neurons. The EM image is at
4 x 4 x 30 nm native resolution; the neuron segmentation at 8 x 8 x 30 nm.

Data is streamed from the BossDB public S3 bucket via cloud-volume and cached locally as
zarr v3 stores (chunk 64^3, shard 512^3, zstd compression) in (z, y, x) axis order.

This dataset is from the publication https://doi.org/10.1016/j.cub.2025.10.001.
Please cite it if you use this dataset in your research.

The dataset is publicly available at https://bossdb.org/project/zhang2025 (DOI 10.60533/BOSS-2025-08G4).
Requires cloud-volume: pip install cloud-volume.
"""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


HYDRA_EM_URL = "precomputed://https://bossdb-open-data.s3.amazonaws.com/zhang2025/image"
HYDRA_SEG_URL = "precomputed://https://bossdb-open-data.s3.amazonaws.com/zhang2025/neurons"

# Pre-defined bounding boxes (nm): (x_min, x_max, y_min, y_max, z_min, z_max).
# Each box is 32768 x 32768 x 18000 nm, placed in the regions with the densest
# neuron annotations (verified by scanning the full volume).
# At default resolution (image_mip=3, seg_mip=2, both 32 x 32 x 30 nm) each box
# is 1024 x 1024 x 600 voxels (~630 MB image, ~2.5 GB neurons).
HYDRA_BOUNDING_BOXES = [
    (131072, 163840, 360448, 393216, 18000, 36000),
    (327680, 360448, 163840, 196608, 18000, 36000),
    (163840, 196608, 294912, 327680, 18000, 36000),
    (196608, 229376, 262144, 294912, 18000, 36000),
]

HYDRA_CHUNK_SHAPE = (64, 128, 128)
HYDRA_SHARD_SHAPE = (128, 512, 512)


def _hydra_bbox_to_str(bbox):
    return hashlib.md5("_".join(str(v) for v in bbox).encode()).hexdigest()[:12]


def _hydra_create_array(root, name, shape, dtype, is_label):
    from zarr.codecs import BloscCodec
    shuffle = "bitshuffle" if (np.issubdtype(dtype, np.integer) and is_label) else "shuffle"
    return root.create_array(
        name,
        shape=shape,
        chunks=HYDRA_CHUNK_SHAPE,
        shards=HYDRA_SHARD_SHAPE,
        dtype=dtype,
        compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
    )


def _hydra_bbox_voxels(cv, x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm):
    scale = np.array(cv.resolution)
    x0 = int(np.floor(x_min_nm / scale[0]))
    x1 = int(np.ceil(x_max_nm / scale[0]))
    y0 = int(np.floor(y_min_nm / scale[1]))
    y1 = int(np.ceil(y_max_nm / scale[1]))
    z0 = int(np.floor(z_min_nm / scale[2]))
    z1 = int(np.ceil(z_max_nm / scale[2]))
    return x0, x1, y0, y1, z0, z1, (z1 - z0, y1 - y0, x1 - x0)


def _hydra_download_to_zarr(cv, ds, x0g, y0g, z0g, name):
    shape = ds.shape  # (z, y, x)
    sz, sy, sx = HYDRA_SHARD_SHAPE

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

    target_dtype = np.dtype(ds.dtype)

    def worker(item):
        (z0_, z1_), (y0_, y1_), (x0_, x1_), (gx0, gx1, gy0, gy1, gz0, gz1) = item
        block = np.asarray(cv[gx0:gx1, gy0:gy1, gz0:gz1])
        if block.ndim == 4:
            block = block[..., 0]
        ds[z0_:z1_, y0_:y1_, x0_:x1_] = block.transpose(2, 1, 0).astype(target_dtype)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading '{name}'", smoothing=0.05):
            fut.result()


def get_hydra_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[float, ...],
    image_mip: int = 3,
    seg_mip: int = 2,
    download: bool = False,
) -> str:
    """Stream and cache one Hydra bounding box as a zarr v3 store.

    The zarr store contains:
      - raw: EM grayscale (uint8, z/y/x)
      - labels: neuron instance segmentation (uint32, z/y/x)

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        bounding_box: Region in nm as (x_min, x_max, y_min, y_max, z_min, z_max).
        image_mip: MIP level for the EM image. Default mip=3 gives 32 x 32 x 30 nm resolution.
        seg_mip: MIP level for the neuron segmentation. Default mip=2 gives 32 x 32 x 30 nm resolution.
        download: Whether to stream and cache the data if not present.

    Returns:
        Filepath to the cached zarr store.
    """
    import zarr

    os.makedirs(str(path), exist_ok=True)
    stem = _hydra_bbox_to_str(bounding_box)
    zarr_path = os.path.join(str(path), f"{stem}.zarr")

    def _complete(zp):
        return os.path.isdir(os.path.join(zp, "raw")) and os.path.isdir(os.path.join(zp, "labels"))

    if _complete(zarr_path):
        return zarr_path
    if not download:
        raise RuntimeError(
            f"No cached data at '{zarr_path}'. Set download=True to stream from BossDB."
        )

    try:
        from cloudvolume import CloudVolume
    except ImportError:
        raise ImportError(
            "The 'cloud-volume' package is required: pip install cloud-volume"
        )

    x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm = bounding_box
    print(f"Streaming Hydra bbox {bounding_box} at image_mip={image_mip}, seg_mip={seg_mip} ...")

    em_cv = CloudVolume(HYDRA_EM_URL, use_https=True, mip=image_mip, progress=False, fill_missing=True)
    seg_cv = CloudVolume(HYDRA_SEG_URL, use_https=True, mip=seg_mip, progress=False, fill_missing=True)

    ex0, ex1, ey0, ey1, ez0, ez1, em_shape = _hydra_bbox_voxels(
        em_cv, x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm
    )
    sx0, sx1, sy0, sy1, sz0, sz1, seg_shape = _hydra_bbox_voxels(
        seg_cv, x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm
    )

    shape = tuple(min(e, s) for e, s in zip(em_shape, seg_shape))

    root = zarr.open_group(zarr_path, mode="a")
    root.attrs["bounding_box_nm"] = list(bounding_box)
    root.attrs["image_mip"] = image_mip
    root.attrs["seg_mip"] = seg_mip

    if "raw" not in root:
        ds_raw = _hydra_create_array(root, "raw", shape, np.dtype("uint8"), is_label=False)
        _hydra_download_to_zarr(em_cv, ds_raw, ex0, ey0, ez0, name="raw")

    if "labels" not in root:
        ds_lbl = _hydra_create_array(root, "labels", shape, np.dtype("uint32"), is_label=True)
        _hydra_download_to_zarr(seg_cv, ds_lbl, sx0, sy0, sz0, name="labels")

    print(f"Cached to {zarr_path} (shape {shape})")
    return zarr_path


def get_hydra_paths(
    path: Union[os.PathLike, str],
    bounding_boxes: Optional[Sequence[Tuple[float, ...]]] = None,
    image_mip: int = 3,
    seg_mip: int = 2,
    download: bool = False,
) -> List[str]:
    """Get paths to cached Hydra zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        bounding_boxes: Bounding boxes in nm (x_min, x_max, y_min, y_max, z_min, z_max).
            Defaults to the four pre-defined boxes covering the densest annotated region.
        image_mip: MIP level for the EM image.
        seg_mip: MIP level for the neuron segmentation.
        download: Whether to stream and cache the data if not present.

    Returns:
        Filepaths to the cached zarr stores.
    """
    boxes = list(bounding_boxes) if bounding_boxes is not None else HYDRA_BOUNDING_BOXES
    return [get_hydra_data(path, bb, image_mip, seg_mip, download) for bb in boxes]


def get_hydra_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_boxes: Optional[Sequence[Tuple[float, ...]]] = None,
    image_mip: int = 3,
    seg_mip: int = 2,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Hydra dataset for neuron instance segmentation in FIB-SEM.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_boxes: Bounding boxes in nm (x_min, x_max, y_min, y_max, z_min, z_max).
            Defaults to the four pre-defined boxes covering the densest annotated region.
        image_mip: MIP level for the EM image. Default mip=3 gives 32 x 32 x 30 nm.
        seg_mip: MIP level for the neuron segmentation. Default mip=2 gives 32 x 32 x 30 nm.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3
    paths = get_hydra_paths(path, bounding_boxes, image_mip, seg_mip, download)

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


def get_hydra_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    bounding_boxes: Optional[Sequence[Tuple[float, ...]]] = None,
    image_mip: int = 3,
    seg_mip: int = 2,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for neuron instance segmentation in the Hydra vulgaris FIB-SEM dataset.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_boxes: Bounding boxes in nm (x_min, x_max, y_min, y_max, z_min, z_max).
            Defaults to the four pre-defined boxes covering the densest annotated region.
        image_mip: MIP level for the EM image. Default mip=3 gives 32 x 32 x 30 nm.
        seg_mip: MIP level for the neuron segmentation. Default mip=2 gives 32 x 32 x 30 nm.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_hydra_dataset(
        path=path,
        patch_shape=patch_shape,
        bounding_boxes=bounding_boxes,
        image_mip=image_mip,
        seg_mip=seg_mip,
        download=download,
        offsets=offsets,
        boundaries=boundaries,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
