"""Wildenberg 2023 dataset for synaptic structure segmentation in 3DEM.

The dataset contains two FIB-SEM volumes from mouse primary visual cortex (V1) layer 4,
acquired at 6 x 6 x 40 nm native resolution. Synaptic structures are annotated at
12 x 12 x 40 nm resolution across three auto-segmentation channels:
- psd: postsynaptic density (binary, uint8)
- vesicle_cloud: presynaptic vesicle cloud (binary, uint8)
- saturated: saturated synapse mask (instance, uint32)

Two experiments are available:
- p105: postnatal day 105 mouse (adult, fully developed cortex)
- p14: postnatal day 14 mouse (early developmental stage)

Data is streamed from the BossDB public S3 bucket via cloud-volume and cached locally as
zarr v3 stores in (z, y, x) axis order.

This dataset is from the publication https://doi.org/10.1038/s41467-023-43088-3.
Please cite it if you use this dataset in your research.

The dataset is publicly available at https://bossdb.org/project/wildenberg2023.
Requires cloud-volume: pip install cloud-volume.
"""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import torch_em
from .. import util


WILDENBERG_S3_BASE = "precomputed://https://bossdb-open-data.s3.amazonaws.com/wildenberg2023"

# Per-experiment metadata: BossDB experiment name, EM channel, annotation bounding box in nm
# (x_min, x_max, y_min, y_max, z_min, z_max) covering the full annotated region.
WILDENBERG_EXPERIMENTS: Dict[str, dict] = {
    "p105": {
        "exp_name": "mouse_v1_l4_p105",
        "em_channel": "em",
        # The p105 EM channel on BossDB has cv_x=physical_y and cv_y=physical_x (axes transposed
        # relative to the annotation channels). The download code corrects for this.
        "em_axes_swapped": True,
        "bbox_nm": (576, 120576, 576, 136512, 160, 36320),
    },
    "p14": {
        "exp_name": "mouse_v1_l4_p14",
        "em_channel": "em_aligned",
        "em_axes_swapped": False,
        "bbox_nm": (0, 80256, 0, 115200, 0, 52320),
    },
}

# (channel_name, numpy_dtype, use_bitshuffle_for_compression)
WILDENBERG_LABEL_CHANNELS: Dict[str, tuple] = {
    "psd": ("psd_autoseg", np.dtype("uint8"), False),
    "vesicle_cloud": ("vesicle_autoseg", np.dtype("uint8"), False),
    "saturated": ("saturated_autoseg", np.dtype("uint32"), True),
}

WILDENBERG_CHUNK_SHAPE = (64, 128, 128)
WILDENBERG_SHARD_SHAPE = (128, 512, 512)


def _wildenberg_bbox_to_str(bbox):
    return hashlib.md5("_".join(str(v) for v in bbox).encode()).hexdigest()[:12]


def _wildenberg_create_array(root, name, shape, dtype, is_label):
    from zarr.codecs import BloscCodec
    shuffle = "bitshuffle" if (np.issubdtype(dtype, np.integer) and is_label) else "shuffle"
    return root.create_array(
        name,
        shape=shape,
        chunks=WILDENBERG_CHUNK_SHAPE,
        shards=WILDENBERG_SHARD_SHAPE,
        dtype=dtype,
        compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
    )


def _wildenberg_bbox_voxels(cv, x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm):
    scale = np.array(cv.resolution)
    x0 = int(np.floor(x_min_nm / scale[0]))
    x1 = int(np.ceil(x_max_nm / scale[0]))
    y0 = int(np.floor(y_min_nm / scale[1]))
    y1 = int(np.ceil(y_max_nm / scale[1]))
    z0 = int(np.floor(z_min_nm / scale[2]))
    z1 = int(np.ceil(z_max_nm / scale[2]))
    return x0, x1, y0, y1, z0, z1, (z1 - z0, y1 - y0, x1 - x0)


def _wildenberg_download_to_zarr(cv, ds, x0g, y0g, z0g, name, swap_xy=False):
    shape = ds.shape  # (z, y, x) in physical space
    sz, sy, sx = WILDENBERG_SHARD_SHAPE

    tasks = []
    for z0_ in range(0, shape[0], sz):
        for y0_ in range(0, shape[1], sy):
            for x0_ in range(0, shape[2], sx):
                z1_ = min(z0_ + sz, shape[0])
                y1_ = min(y0_ + sy, shape[1])
                x1_ = min(x0_ + sx, shape[2])
                gz0, gz1 = z0g + z0_, z0g + z1_
                if swap_xy:
                    # em cv_x=physical_y, cv_y=physical_x: map physical y→cv_x, physical x→cv_y
                    gx0, gx1 = y0g + y0_, y0g + y1_
                    gy0, gy1 = x0g + x0_, x0g + x1_
                else:
                    gx0, gx1 = x0g + x0_, x0g + x1_
                    gy0, gy1 = y0g + y0_, y0g + y1_
                tasks.append(((z0_, z1_), (y0_, y1_), (x0_, x1_), (gx0, gx1, gy0, gy1, gz0, gz1)))

    target_dtype = np.dtype(ds.dtype)
    # swap_xy: block axes are (phys_y, phys_x, phys_z) → transpose(2,0,1) → (z,y,x)
    # normal:  block axes are (phys_x, phys_y, phys_z) → transpose(2,1,0) → (z,y,x)
    transpose_order = (2, 0, 1) if swap_xy else (2, 1, 0)

    def worker(item):
        (z0_, z1_), (y0_, y1_), (x0_, x1_), (gx0, gx1, gy0, gy1, gz0, gz1) = item
        block = np.asarray(cv[gx0:gx1, gy0:gy1, gz0:gz1])
        if block.ndim == 4:
            block = block[..., 0]
        ds[z0_:z1_, y0_:y1_, x0_:x1_] = block.transpose(*transpose_order).astype(target_dtype)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading '{name}'", smoothing=0.05):
            fut.result()


def get_wildenberg_data(
    path: Union[os.PathLike, str],
    experiment: Literal["p105", "p14"],
    label_choice: Literal["psd", "vesicle_cloud", "saturated"],
    bounding_box: Optional[Tuple[float, ...]] = None,
    em_mip: int = 1,
    seg_mip: int = 0,
    download: bool = False,
) -> str:
    """Stream and cache one Wildenberg experiment as a zarr v3 store.

    The zarr store contains:
      - raw: EM grayscale (uint8, z/y/x)
      - labels: synaptic annotation (uint8 for psd/vesicle_cloud, uint32 for saturated, z/y/x)

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        experiment: Which experiment to load. Either 'p105' (adult) or 'p14' (developing).
        label_choice: Which annotation channel to use. One of 'psd', 'vesicle_cloud', or 'saturated'.
        bounding_box: Region in nm as (x_min, x_max, y_min, y_max, z_min, z_max).
            Defaults to the full annotation extent of the chosen experiment.
        em_mip: MIP level for the EM image. Default mip=1 gives 12 x 12 x 40 nm resolution.
        seg_mip: MIP level for the annotation. Default mip=0 gives 12 x 12 x 40 nm resolution.
        download: Whether to stream and cache the data if not present.

    Returns:
        Filepath to the cached zarr store.
    """
    import zarr

    if experiment not in WILDENBERG_EXPERIMENTS:
        raise ValueError(f"Invalid experiment: '{experiment}'. Choose from {list(WILDENBERG_EXPERIMENTS.keys())}.")
    if label_choice not in WILDENBERG_LABEL_CHANNELS:
        raise ValueError(
            f"Invalid label_choice: '{label_choice}'. Choose from {list(WILDENBERG_LABEL_CHANNELS.keys())}."
        )

    os.makedirs(str(path), exist_ok=True)
    bbox = bounding_box if bounding_box is not None else WILDENBERG_EXPERIMENTS[experiment]["bbox_nm"]
    bbox_hash = _wildenberg_bbox_to_str(bbox)
    zarr_path = os.path.join(str(path), f"{experiment}_{label_choice}_{bbox_hash}.zarr")

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
        raise ImportError("The 'cloud-volume' package is required: pip install cloud-volume")

    exp_info = WILDENBERG_EXPERIMENTS[experiment]
    exp_name = exp_info["exp_name"]
    em_channel = exp_info["em_channel"]
    em_axes_swapped = exp_info.get("em_axes_swapped", False)
    x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm = bbox

    label_channel, label_dtype, label_compress = WILDENBERG_LABEL_CHANNELS[label_choice]

    print(f"Streaming Wildenberg2023 {experiment}/{label_choice} at em_mip={em_mip}, seg_mip={seg_mip} ...")

    em_url = f"{WILDENBERG_S3_BASE}/{exp_name}/{em_channel}"
    seg_url = f"{WILDENBERG_S3_BASE}/{exp_name}/{label_channel}"

    em_cv = CloudVolume(em_url, use_https=True, mip=em_mip, progress=False, fill_missing=True)
    seg_cv = CloudVolume(seg_url, use_https=True, mip=seg_mip, progress=False, fill_missing=True)

    # Clip the requested nm bbox to the intersection of both volumes' actual extents so that
    # the EM and label arrays start at the same physical coordinate (avoiding spatial offsets).
    em_scale = np.array(em_cv.resolution, dtype=float)
    seg_scale = np.array(seg_cv.resolution, dtype=float)
    em_bb = em_cv.meta.bbox(em_mip)
    seg_bb = seg_cv.meta.bbox(seg_mip)
    em_min_nm = np.array(em_bb.minpt[:3], dtype=float) * em_scale
    em_max_nm = np.array(em_bb.maxpt[:3], dtype=float) * em_scale
    seg_min_nm = np.array(seg_bb.minpt[:3], dtype=float) * seg_scale
    seg_max_nm = np.array(seg_bb.maxpt[:3], dtype=float) * seg_scale

    cx_min = max(x_min_nm, float(em_min_nm[0]), float(seg_min_nm[0]))
    cx_max = min(x_max_nm, float(em_max_nm[0]), float(seg_max_nm[0]))
    cy_min = max(y_min_nm, float(em_min_nm[1]), float(seg_min_nm[1]))
    cy_max = min(y_max_nm, float(em_max_nm[1]), float(seg_max_nm[1]))
    cz_min = max(z_min_nm, float(em_min_nm[2]), float(seg_min_nm[2]))
    cz_max = min(z_max_nm, float(em_max_nm[2]), float(seg_max_nm[2]))

    ex0, ex1, ey0, ey1, ez0, ez1, em_shape = _wildenberg_bbox_voxels(
        em_cv, cx_min, cx_max, cy_min, cy_max, cz_min, cz_max
    )
    sx0, sx1, sy0, sy1, sz0, sz1, seg_shape = _wildenberg_bbox_voxels(
        seg_cv, cx_min, cx_max, cy_min, cy_max, cz_min, cz_max
    )

    shape = tuple(min(e, s) for e, s in zip(em_shape, seg_shape))

    root = zarr.open_group(zarr_path, mode="a")
    root.attrs["experiment"] = experiment
    root.attrs["label_choice"] = label_choice
    root.attrs["bounding_box_nm"] = list(bbox)
    root.attrs["em_mip"] = em_mip
    root.attrs["seg_mip"] = seg_mip

    if "raw" not in root:
        ds_raw = _wildenberg_create_array(root, "raw", shape, np.dtype("uint8"), is_label=False)
        _wildenberg_download_to_zarr(em_cv, ds_raw, ex0, ey0, ez0, name="raw", swap_xy=em_axes_swapped)

    if "labels" not in root:
        ds_lbl = _wildenberg_create_array(root, "labels", shape, label_dtype, is_label=label_compress)
        _wildenberg_download_to_zarr(seg_cv, ds_lbl, sx0, sy0, sz0, name="labels")

    print(f"Cached to {zarr_path} (shape {shape})")
    return zarr_path


def get_wildenberg_paths(
    path: Union[os.PathLike, str],
    experiments: Optional[Sequence[str]] = None,
    label_choice: Literal["psd", "vesicle_cloud", "saturated"] = "psd",
    bounding_box: Optional[Tuple[float, ...]] = None,
    em_mip: int = 1,
    seg_mip: int = 0,
    download: bool = False,
) -> List[str]:
    """Get paths to cached Wildenberg zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        experiments: Experiments to load. Defaults to both ('p105', 'p14').
        label_choice: Which annotation channel to use. One of 'psd', 'vesicle_cloud', or 'saturated'.
        bounding_box: Region in nm as (x_min, x_max, y_min, y_max, z_min, z_max).
            Defaults to the full annotation extent per experiment.
        em_mip: MIP level for the EM image.
        seg_mip: MIP level for the annotation.
        download: Whether to stream and cache the data if not present.

    Returns:
        Filepaths to the cached zarr stores.
    """
    exps = list(experiments) if experiments is not None else list(WILDENBERG_EXPERIMENTS.keys())
    return [get_wildenberg_data(path, exp, label_choice, bounding_box, em_mip, seg_mip, download) for exp in exps]


def get_wildenberg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    experiments: Optional[Sequence[str]] = None,
    label_choice: Literal["psd", "vesicle_cloud", "saturated"] = "psd",
    bounding_box: Optional[Tuple[float, ...]] = None,
    em_mip: int = 1,
    seg_mip: int = 0,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Wildenberg 2023 dataset for synaptic structure segmentation in 3DEM.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        experiments: Experiments to load. Defaults to both ('p105', 'p14').
        label_choice: Which annotation channel to use. 'psd' for postsynaptic density,
            'vesicle_cloud' for presynaptic vesicle cloud, or 'saturated' for instance-labeled saturated synapses.
        bounding_box: Region in nm as (x_min, x_max, y_min, y_max, z_min, z_max).
            Defaults to the full annotation extent per experiment.
        em_mip: MIP level for the EM image. Default mip=1 gives 12 x 12 x 40 nm.
        seg_mip: MIP level for the annotation. Default mip=0 gives 12 x 12 x 40 nm.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation (only applied when label_choice='saturated').
        boundaries: Whether to compute boundaries (only applied when label_choice='saturated').
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3
    paths = get_wildenberg_paths(path, experiments, label_choice, bounding_box, em_mip, seg_mip, download)

    if label_choice == "saturated":
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


def get_wildenberg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    experiments: Optional[Sequence[str]] = None,
    label_choice: Literal["psd", "vesicle_cloud", "saturated"] = "psd",
    bounding_box: Optional[Tuple[float, ...]] = None,
    em_mip: int = 1,
    seg_mip: int = 0,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for synaptic structure segmentation in Wildenberg 2023 3DEM data.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape (z, y, x) to use for training.
        experiments: Experiments to load. Defaults to both ('p105', 'p14').
        label_choice: Which annotation channel to use. 'psd' for postsynaptic density,
            'vesicle_cloud' for presynaptic vesicle cloud, or 'saturated' for instance-labeled saturated synapses.
        bounding_box: Region in nm as (x_min, x_max, y_min, y_max, z_min, z_max).
            Defaults to the full annotation extent per experiment.
        em_mip: MIP level for the EM image. Default mip=1 gives 12 x 12 x 40 nm.
        seg_mip: MIP level for the annotation. Default mip=0 gives 12 x 12 x 40 nm.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation (only applied when label_choice='saturated').
        boundaries: Whether to compute boundaries (only applied when label_choice='saturated').
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_wildenberg_dataset(
        path=path,
        patch_shape=patch_shape,
        experiments=experiments,
        label_choice=label_choice,
        bounding_box=bounding_box,
        em_mip=em_mip,
        seg_mip=seg_mip,
        download=download,
        offsets=offsets,
        boundaries=boundaries,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
