"""Zebrafinch Area X datasets for neuron and organelle segmentation in 3DEM.

Two FIB-SEM volumes of adult male zebra finch (Taeniopygia guttata) area X are
available, both from the Kornfeld lab:

- j0251: 10 x 10 x 25 nm native resolution, full extent ~256 x 256 x 384 µm.
  Labels: neuron instance segmentation (~4.26 M neurons) and endoplasmic reticulum.
  Cell-type labels (17 types: MSN, GPe, GPi, HVC axons, interneurons, etc.) and
  synapse coordinates are available via the REST API at https://syconn.esc.mpcdf.mpg.de.
- j0126: 10 x 10 x 20 nm native resolution, full extent ~107 x 109 x 114 µm.
  Labels: neuron instance segmentation only.

Data is streamed from the Kornfeld lab public server via cloud-volume and cached
locally as zarr v3 stores in (z, y, x) axis order.

This dataset is from the publication https://doi.org/10.1101/2025.10.25.684569.
Please cite it if you use this dataset in your research.

The dataset is publicly available at https://syconn.esc.mpcdf.mpg.de.
Requires cloud-volume: pip install cloud-volume.
"""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import torch_em
from .. import util


J0251_BASE_URL = (
    "precomputed://https://syconn.esc.mpcdf.mpg.de"
    "/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822"
)
J0126_BASE_URL = "precomputed://https://syconn.esc.mpcdf.mpg.de"

ZEBRAFINCH_DATASETS = {
    "j0251": {
        "em_url": f"{J0251_BASE_URL}/image",
        "seg_url": f"{J0251_BASE_URL}/segmentation",
        "er_url": f"{J0251_BASE_URL}/er",
        # Full extent ~256 x 256 x 384 µm at 10 x 10 x 25 nm native resolution.
        "bbox_nm": (0, 271190, 0, 273500, 0, 387350),
    },
    "j0126": {
        "em_url": f"{J0126_BASE_URL}/j0126/volume/image",
        "seg_url": f"{J0126_BASE_URL}/volume/segmentation",
        "er_url": None,
        # Full extent ~107 x 109 x 114 µm at 10 x 10 x 20 nm native resolution.
        "bbox_nm": (0, 106640, 0, 109130, 0, 114000),
    },
}

ZEBRAFINCH_CHUNK_SHAPE = (64, 128, 128)
ZEBRAFINCH_SHARD_SHAPE = (128, 512, 512)


def _zebrafinch_bbox_to_str(bbox):
    return hashlib.md5("_".join(str(v) for v in bbox).encode()).hexdigest()[:12]


def _zebrafinch_create_array(root, name, shape, dtype, is_label):
    from zarr.codecs import BloscCodec
    shuffle = "bitshuffle" if (np.issubdtype(dtype, np.integer) and is_label) else "shuffle"
    return root.create_array(
        name,
        shape=shape,
        chunks=ZEBRAFINCH_CHUNK_SHAPE,
        shards=ZEBRAFINCH_SHARD_SHAPE,
        dtype=dtype,
        compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
    )


def _zebrafinch_bbox_voxels(cv, x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm):
    scale = np.array(cv.resolution)
    x0 = int(np.floor(x_min_nm / scale[0]))
    x1 = int(np.ceil(x_max_nm / scale[0]))
    y0 = int(np.floor(y_min_nm / scale[1]))
    y1 = int(np.ceil(y_max_nm / scale[1]))
    z0 = int(np.floor(z_min_nm / scale[2]))
    z1 = int(np.ceil(z_max_nm / scale[2]))
    return x0, x1, y0, y1, z0, z1, (z1 - z0, y1 - y0, x1 - x0)


def _zebrafinch_download_to_zarr(cv, ds, x0g, y0g, z0g, name):
    shape = ds.shape  # (z, y, x)
    sz, sy, sx = ZEBRAFINCH_SHARD_SHAPE

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


def get_zebrafinch_data(
    path: Union[os.PathLike, str],
    bounding_box: Optional[Tuple[float, ...]] = None,
    mip: int = 0,
    dataset: Literal["j0251", "j0126"] = "j0251",
    download: bool = False,
) -> str:
    """Stream and cache a region of a zebrafinch dataset as a zarr v3 store.

    The zarr store contains:
      - raw: EM grayscale (uint8, z/y/x)
      - labels: neuron instance segmentation (uint64, z/y/x)
      - er: endoplasmic reticulum instance segmentation (uint64, z/y/x) - j0251 only.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        bounding_box: Region in nm as (x_min, x_max, y_min, y_max, z_min, z_max).
            Defaults to the full volume extent for the chosen dataset.
        mip: MIP level for both EM and segmentation. Default mip=0 gives native resolution
            (10 x 10 x 25 nm for j0251, 10 x 10 x 20 nm for j0126).
        dataset: Which specimen to use, either "j0251" or "j0126".
        download: Whether to stream and cache the data if not present.

    Returns:
        Filepath to the cached zarr store.
    """
    import zarr

    ds_info = ZEBRAFINCH_DATASETS[dataset]
    os.makedirs(str(path), exist_ok=True)
    bbox = bounding_box if bounding_box is not None else ds_info["bbox_nm"]
    bbox_hash = _zebrafinch_bbox_to_str(bbox)
    zarr_path = os.path.join(str(path), f"{dataset}_mip{mip}_{bbox_hash}.zarr")

    arrays_needed = ["raw", "labels"] + (["er"] if ds_info["er_url"] is not None else [])
    root = zarr.open_group(zarr_path, mode="a")
    missing = [k for k in arrays_needed if k not in root]
    if not missing:
        return zarr_path
    if not download:
        raise RuntimeError(
            f"No cached data at '{zarr_path}'. Set download=True to stream from the Kornfeld lab server."
        )

    try:
        from cloudvolume import CloudVolume
    except ImportError:
        raise ImportError("The 'cloud-volume' package is required: pip install cloud-volume")

    x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm = bbox
    print(f"Streaming zebrafinch {dataset} at mip={mip} ...")

    cv_kwargs = dict(use_https=True, mip=mip, progress=False, fill_missing=True, provenance={})
    em_cv = CloudVolume(ds_info["em_url"], **cv_kwargs)
    seg_cv = CloudVolume(ds_info["seg_url"], **cv_kwargs)

    ex0, ex1, ey0, ey1, ez0, ez1, em_shape = _zebrafinch_bbox_voxels(
        em_cv, x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm
    )
    sx0, sx1, sy0, sy1, sz0, sz1, seg_shape = _zebrafinch_bbox_voxels(
        seg_cv, x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm
    )
    shape = tuple(min(e, s) for e, s in zip(em_shape, seg_shape))

    root.attrs["bounding_box_nm"] = list(bbox)
    root.attrs["mip"] = mip

    if "raw" not in root:
        ds_raw = _zebrafinch_create_array(root, "raw", shape, np.dtype("uint8"), is_label=False)
        _zebrafinch_download_to_zarr(em_cv, ds_raw, ex0, ey0, ez0, name="raw")

    if "labels" not in root:
        ds_lbl = _zebrafinch_create_array(root, "labels", shape, np.dtype("uint64"), is_label=True)
        _zebrafinch_download_to_zarr(seg_cv, ds_lbl, sx0, sy0, sz0, name="labels")

    if "er" not in root and ds_info["er_url"] is not None:
        er_cv = CloudVolume(ds_info["er_url"], **cv_kwargs)
        rx0, rx1, ry0, ry1, rz0, rz1, er_shape = _zebrafinch_bbox_voxels(
            er_cv, x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm
        )
        shape_er = tuple(min(e, r) for e, r in zip(shape, er_shape))
        ds_er = _zebrafinch_create_array(root, "er", shape_er, np.dtype("uint64"), is_label=True)
        _zebrafinch_download_to_zarr(er_cv, ds_er, rx0, ry0, rz0, name="er")

    print(f"Cached to {zarr_path} (shape {shape})")
    return zarr_path


def get_zebrafinch_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_box: Optional[Tuple[float, ...]] = None,
    mip: int = 0,
    dataset: Literal["j0251", "j0126"] = "j0251",
    label_choice: Literal["neurons", "er"] = "neurons",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get a zebrafinch dataset for neuron or organelle segmentation.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_box: Region in nm as (x_min, x_max, y_min, y_max, z_min, z_max).
            Defaults to the full volume extent for the chosen dataset.
        mip: MIP level for both EM and segmentation. Default mip=0 gives native resolution
            (10 x 10 x 25 nm for j0251, 10 x 10 x 20 nm for j0126).
        dataset: Which specimen to use, either "j0251" or "j0126".
        label_choice: Which segmentation to use as target. Either "neurons" or "er".
            "er" is only available for j0251.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3
    if label_choice == "er" and ZEBRAFINCH_DATASETS[dataset]["er_url"] is None:
        raise ValueError(f"label_choice='er' is not available for dataset='{dataset}'")
    zarr_path = get_zebrafinch_data(path, bounding_box, mip, dataset, download)

    label_key = "labels" if label_choice == "neurons" else "er"

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=zarr_path,
        raw_key="raw",
        label_paths=zarr_path,
        label_key=label_key,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_zebrafinch_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    bounding_box: Optional[Tuple[float, ...]] = None,
    mip: int = 0,
    dataset: Literal["j0251", "j0126"] = "j0251",
    label_choice: Literal["neurons", "er"] = "neurons",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for neuron or organelle segmentation in a zebrafinch dataset.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_box: Region in nm as (x_min, x_max, y_min, y_max, z_min, z_max).
            Defaults to the full volume extent for the chosen dataset.
        mip: MIP level for both EM and segmentation. Default mip=0 gives native resolution
            (10 x 10 x 25 nm for j0251, 10 x 10 x 20 nm for j0126).
        dataset: Which specimen to use, either "j0251" or "j0126".
        label_choice: Which segmentation to use as target. Either "neurons" or "er".
            "er" is only available for j0251.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_zebrafinch_dataset(
        path=path,
        patch_shape=patch_shape,
        bounding_box=bounding_box,
        mip=mip,
        dataset=dataset,
        label_choice=label_choice,
        download=download,
        offsets=offsets,
        boundaries=boundaries,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
