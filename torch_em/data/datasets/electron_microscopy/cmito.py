"""Cmito dataset for dense mitochondria segmentation in whole-animal C. elegans EM volumes.

The dataset provides dense mitochondria instance masks for five developmental stages of
C. elegans, built on public whole-animal EM connectome volumes:

- l1, l2, l3, adult: from Witvliet et al. 2021 (https://doi.org/10.1038/s41586-021-03778-8),
  CC-BY-4.0, deposited at https://zenodo.org/records/5637219 and https://bossdb.org/project/witvliet2020.
- dauer: from Yim et al. 2024 (https://doi.org/10.1038/s41467-024-45943-3), whose article text
  is CC-BY-4.0. The mitochondria mask bucket for this stage lives in a separate GCS project
  (`gnd-dauer1`) than the other four stages, and no repository-level data license could be
  confirmed for it independently of the article license. Treat "dauer" as license-uncertain
  until a repository landing page or downloadable asset states its data license explicitly.

Raw EM is streamed from BossDB (S3) at a mip level chosen per stage so that it spatially matches
the coarser, separately hosted mitochondria mask (on Google Cloud Storage, at native resolution).
Both are cached locally as zarr v3 stores in (z, y, x) axis order.

This dataset is from the publication https://doi.org/10.1101/2024.07.19.604219.
Please cite it if you use this dataset in your research.

Corresponding authors: J. Alexander Bae (jabae@snu.ac.kr), Junho Lee (elegans@snu.ac.kr).
Original tool repository: https://github.com/jabae/Cmito.
Requires cloud-volume: pip install cloud-volume.
"""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import torch_em
from .. import util


CMITO_STAGES = {
    "l1": {
        "raw_url": "precomputed://https://bossdb-open-data.s3.amazonaws.com/witvliet2020/Dataset_2/em",
        "mask_url": "precomputed://https://storage.googleapis.com/gnd-neuroglancer/witvliet/dataset2/mito_seg_v3",
        # Raw mip3 (0.64 nm x 8 = 5.12 nm xy) spatially matches the mask's native 5.12/5.12/50 nm.
        "raw_mip": 3,
        # Full extent in nm, from the mip0 shape (26624, 22016, 368) x resolution (0.64, 0.64, 50).
        "bbox_nm": (0, 17039, 0, 14090, 0, 18400),
    },
    "l2": {
        "raw_url": "precomputed://https://bossdb-open-data.s3.amazonaws.com/witvliet2020/Dataset_5/em",
        "mask_url": "precomputed://https://storage.googleapis.com/gnd-neuroglancer/witvliet/dataset5/mito_seg_v4",
        # Raw mip2 (2 nm x 4 = 8 nm xy) spatially matches the mask's native 8/8/30 nm.
        "raw_mip": 2,
        "bbox_nm": (0, 27648, 0, 22528, 0, 25920),
    },
    "l3": {
        "raw_url": "precomputed://https://bossdb-open-data.s3.amazonaws.com/witvliet2020/Dataset_6/em",
        "mask_url": "precomputed://https://storage.googleapis.com/gnd-neuroglancer/witvliet/dataset6/mito_seg_v4",
        # Raw mip3 (8x downsample) spatially matches the mask, despite the mask's reported
        # 8 nm nominal resolution not being an exact 8x multiple of the raw 0.768 nm mip0
        # pixel size; shapes (4544 x 4288) confirm the 8x factor is the correct match.
        "raw_mip": 3,
        "bbox_nm": (0, 27918, 0, 26345, 0, 21600),
    },
    "adult": {
        "raw_url": "precomputed://https://bossdb-open-data.s3.amazonaws.com/witvliet2020/Dataset_8/em",
        "mask_url": "precomputed://https://storage.googleapis.com/gnd-neuroglancer/witvliet/dataset8/mito_seg_v3",
        # Raw mip3 (2 nm x 8 = 16 nm xy) spatially matches the mask's native 16/16/30 nm.
        "raw_mip": 3,
        "bbox_nm": (0, 79872, 0, 44032, 0, 21120),
    },
    "dauer": {
        "raw_url": "precomputed://https://bossdb-open-data.s3.amazonaws.com/yim_choe_bae2023/dauer1_364/em/em",
        # NOTE: different GCS bucket/project than the other four stages.
        "mask_url": "precomputed://https://storage.googleapis.com/gnd-dauer1/dauer1_364/mito_seg_v4",
        # Raw mip3 (1 nm x 8 = 8 nm xy) spatially matches the mask's native 8/8/50 nm.
        "raw_mip": 3,
        "bbox_nm": (0, 9800, 0, 9792, 0, 18200),
    },
}

CMITO_CHUNK_SHAPE = (64, 128, 128)
CMITO_SHARD_SHAPE = (128, 512, 512)


def _cmito_bbox_to_str(bbox):
    return hashlib.md5("_".join(str(v) for v in bbox).encode()).hexdigest()[:12]


def _cmito_create_array(root, name, shape, dtype, is_label):
    from zarr.codecs import BloscCodec
    shuffle = "bitshuffle" if (np.issubdtype(dtype, np.integer) and is_label) else "shuffle"
    return root.create_array(
        name,
        shape=shape,
        chunks=CMITO_CHUNK_SHAPE,
        shards=CMITO_SHARD_SHAPE,
        dtype=dtype,
        compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
    )


def _cmito_bbox_voxels(cv, x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm):
    scale = np.array(cv.resolution)
    x0 = int(np.floor(x_min_nm / scale[0]))
    x1 = int(np.ceil(x_max_nm / scale[0]))
    y0 = int(np.floor(y_min_nm / scale[1]))
    y1 = int(np.ceil(y_max_nm / scale[1]))
    z0 = int(np.floor(z_min_nm / scale[2]))
    z1 = int(np.ceil(z_max_nm / scale[2]))
    return x0, x1, y0, y1, z0, z1, (z1 - z0, y1 - y0, x1 - x0)


def _cmito_download_to_zarr(cv, ds, x0g, y0g, z0g, name):
    shape = ds.shape  # (z, y, x)
    sz, sy, sx = CMITO_SHARD_SHAPE

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


def get_cmito_data(
    path: Union[os.PathLike, str],
    stage: Literal["l1", "l2", "l3", "adult", "dauer"],
    bounding_box: Optional[Tuple[float, ...]] = None,
    download: bool = False,
) -> str:
    """Stream and cache one Cmito developmental-stage volume as a zarr v3 store.

    The zarr store contains:
      - raw: EM grayscale (uint8, z/y/x), at the mip level that spatially matches the mask.
      - labels: mitochondria instance segmentation (uint16, z/y/x), at native mask resolution.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        stage: The developmental stage to use. One of 'l1', 'l2', 'l3', 'adult', 'dauer'.
            The 'dauer' stage's mask bucket has no independently confirmed data license
            (see the module docstring); use it with that caveat in mind.
        bounding_box: Region in nm as (x_min, x_max, y_min, y_max, z_min, z_max).
            Defaults to the full volume extent for the chosen stage.
        download: Whether to stream and cache the data if not present.

    Returns:
        Filepath to the cached zarr store.
    """
    import zarr

    if stage not in CMITO_STAGES:
        raise ValueError(f"Invalid stage: '{stage}'. Choose from {list(CMITO_STAGES.keys())}.")

    stage_info = CMITO_STAGES[stage]
    os.makedirs(str(path), exist_ok=True)
    bbox = bounding_box if bounding_box is not None else stage_info["bbox_nm"]
    bbox_hash = _cmito_bbox_to_str(bbox)
    zarr_path = os.path.join(str(path), f"{stage}_{bbox_hash}.zarr")

    def _complete(zp):
        return os.path.isdir(os.path.join(zp, "raw")) and os.path.isdir(os.path.join(zp, "labels"))

    if _complete(zarr_path):
        return zarr_path
    if not download:
        raise RuntimeError(
            f"No cached data at '{zarr_path}'. Set download=True to stream from BossDB and GCS."
        )

    try:
        from cloudvolume import CloudVolume
    except ImportError:
        raise ImportError("The 'cloud-volume' package is required: pip install cloud-volume")

    x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm = bbox
    raw_mip = stage_info["raw_mip"]
    print(f"Streaming Cmito stage='{stage}' at raw_mip={raw_mip} ...")

    raw_cv = CloudVolume(
        stage_info["raw_url"], use_https=True, mip=raw_mip, progress=False, fill_missing=True,
    )
    mask_cv = CloudVolume(stage_info["mask_url"], mip=0, progress=False, fill_missing=True)

    rx0, rx1, ry0, ry1, rz0, rz1, raw_shape = _cmito_bbox_voxels(
        raw_cv, x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm
    )
    mx0, mx1, my0, my1, mz0, mz1, mask_shape = _cmito_bbox_voxels(
        mask_cv, x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm
    )
    shape = tuple(min(r, m) for r, m in zip(raw_shape, mask_shape))

    root = zarr.open_group(zarr_path, mode="a")
    root.attrs["stage"] = stage
    root.attrs["bounding_box_nm"] = list(bbox)
    root.attrs["raw_mip"] = raw_mip

    if "raw" not in root:
        ds_raw = _cmito_create_array(root, "raw", shape, np.dtype("uint8"), is_label=False)
        _cmito_download_to_zarr(raw_cv, ds_raw, rx0, ry0, rz0, name="raw")

    if "labels" not in root:
        ds_lbl = _cmito_create_array(root, "labels", shape, np.dtype("uint16"), is_label=True)
        _cmito_download_to_zarr(mask_cv, ds_lbl, mx0, my0, mz0, name="labels")

    print(f"Cached to {zarr_path} (shape {shape})")
    return zarr_path


def get_cmito_paths(
    path: Union[os.PathLike, str],
    stages: Optional[Sequence[str]] = None,
    bounding_box: Optional[Tuple[float, ...]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to cached Cmito zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        stages: Developmental stages to load. Defaults to all five stages.
        bounding_box: Region in nm as (x_min, x_max, y_min, y_max, z_min, z_max), applied to
            every requested stage. Defaults to each stage's own full volume extent.
        download: Whether to stream and cache the data if not present.

    Returns:
        Filepaths to the cached zarr stores.
    """
    stages_ = list(stages) if stages is not None else list(CMITO_STAGES.keys())
    return [get_cmito_data(path, stage, bounding_box, download) for stage in stages_]


def get_cmito_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    stage: Union[str, Sequence[str]] = "l1",
    bounding_box: Optional[Tuple[float, ...]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Cmito dataset for mitochondria instance segmentation in whole-animal C. elegans EM.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        stage: The developmental stage(s) to use, one or several of 'l1', 'l2', 'l3', 'adult',
            'dauer'. The 'dauer' stage's mask bucket has no independently confirmed data license
            (see the module docstring); use it with that caveat in mind.
        bounding_box: Region in nm as (x_min, x_max, y_min, y_max, z_min, z_max), applied to
            every requested stage. Defaults to each stage's own full volume extent.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3
    stages = [stage] if isinstance(stage, str) else list(stage)
    paths = get_cmito_paths(path, stages, bounding_box, download)

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


def get_cmito_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    stage: Union[str, Sequence[str]] = "l1",
    bounding_box: Optional[Tuple[float, ...]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for mitochondria instance segmentation in whole-animal C. elegans EM.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape (z, y, x) to use for training.
        stage: The developmental stage(s) to use, one or several of 'l1', 'l2', 'l3', 'adult',
            'dauer'. The 'dauer' stage's mask bucket has no independently confirmed data license
            (see the module docstring); use it with that caveat in mind.
        bounding_box: Region in nm as (x_min, x_max, y_min, y_max, z_min, z_max), applied to
            every requested stage. Defaults to each stage's own full volume extent.
        download: Whether to stream and cache data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_cmito_dataset(
        path=path,
        patch_shape=patch_shape,
        stage=stage,
        bounding_box=bounding_box,
        download=download,
        offsets=offsets,
        boundaries=boundaries,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
