"""This dataset provides nucleus segmentation masks for 30 volumes from the public
"OpenOrganelle" `janelia-cosem-datasets` S3 bucket (the same bucket used by `cellmap.py` and
`janelia_nucleus.py`), none of which are covered by either of those existing loaders. All 30 are
released under CC0-1.0, except `jrc_mus-thymus-1` and `aic_desmosome-3` whose license and paper
could not be independently verified (`aic_desmosome-3` is an Allen Institute dataset, so the
Janelia CC0-1.0 convention documented for the rest does not apply to it).

Three label folder naming conventions are used across the bucket, all distinct from
`janelia_nucleus.py`'s:

- `labels/inference/nucleus_seg/{level}` (`jrc_ctl-id8-2`, `jrc_choroid-plexus-2`, `jrc_dauer-larva`).
- `labels/inference/segmentations/nuc/{level}` (most of the remaining 27 datasets).
- `labels/inference/segmentations/nucleus/{level}` (`jrc_hum-airway-14953vc` only).
- `labels/inference/segmentations/nuc_mem/{level}` (`jrc_mus-liver-4/5/6`) - these three are
  nuclear MEMBRANE labels, a distinct annotation target from a nucleus mask/instance segmentation,
  tracked via `DATASETS[name]["label_target"]`.

Label and raw data are usually released at an exactly matching multiscale pyramid level (no
resampling, same axis order). Three exceptions are handled: `jrc_dauer-larva` and
`jrc_hela-h89-1/2` have no exact-scale match and are resampled onto the label grid via
`scipy.ndimage.zoom`; `jrc_choroid-plexus-2` and `jrc_mus-epididymis-1/2` store one of the two
arrays with a reversed axis order, corrected via `np.transpose`.

Datasets whose full matched-resolution raw+label pair is small (roughly under 5 GB) are
downloaded whole, following `janelia_nucleus.py`'s pattern. The rest (full volumes ranging from
tens of GB to several TB) are downloaded as one fixed-size representative crop, auto-detected
from a non-empty region of the coarsest available label pyramid level, following `cellmap.py`'s
per-crop pattern - see `DATASETS[name]["full_array"]`.

`jrc_mosquito-stylet-6` is the only dataset whose raw and label arrays are stored in zarr v3 with
sharding (1024^3 outer shards, 64^3 inner chunks); `_open_remote_zarr` uses `zarr.storage.
FsspecStore` rather than `fsspec.get_mapper` specifically so that a crop only pulls the individual
inner chunks it needs via ranged reads, not whole ~1 GB shard files.

Please cite the CellMap project (https://www.janelia.org/project-team/cellmap) if you use this
data (except `aic_desmosome-3`, an Allen Institute for Cell Science dataset).
"""

import os
from typing import List, Optional, Tuple, Union

import numpy as np

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


BUCKET_URL = "https://janelia-cosem-datasets.s3.amazonaws.com/"

# dataset name -> recon folder, em array name, label subpath (relative to "labels/inference/"),
# label pyramid level, label kind, label target, axis to reverse ("none", "label", or "raw"),
# and whether the full matched-resolution pair is downloaded whole (True) or as one fixed-size
# crop auto-detected from a non-empty region (False).
# label_kind: "binary" = label array only ever takes values {0, 1}; "instance" = per-nucleus IDs.
# label_target: "nucleus" (a nucleus mask/instance segmentation) or "nuclear_membrane" (a nuclear
# envelope surface label, a distinct annotation target).
DATASETS = {
    "jrc_ctl-id8-2": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "nucleus_seg", "label_level": "s0",
        "label_kind": "binary", "label_target": "nucleus", "axis_reverse": "none", "full_array": False,
        "bounding_box": ((3824, 4048), (448, 672), (4064, 4288)),
    },
    "jrc_choroid-plexus-2": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "nucleus_seg", "label_level": "s1",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "label", "full_array": False,
        "bounding_box": ((780, 1100), (2220, 2540), (620, 940)),
    },
    "jrc_dauer-larva": {
        "recon": "recon-1", "em_name": "tem-uint8", "label_subpath": "nucleus_seg", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": False,
        "bounding_box": ((130, 330), (1804, 2004), (11724, 11924)),
    },
    "jrc_ccl81-covid-1": {
        "recon": "recon-1", "em_name": "fibsem-uint16", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": False,
    },
    "jrc_cos7-11": {
        "recon": "recon-1", "em_name": "fibsem-uint16", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": False,
    },
    "jrc_ctl-id8-3": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": False,
    },
    "jrc_ctl-id8-4": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": False,
    },
    "jrc_ctl-id8-5": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": False,
    },
    "jrc_fly-acc-calyx-1": {
        "recon": "recon-1", "em_name": "fibsem-uint16", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": True,
    },
    "jrc_fly-fsb-1": {
        "recon": "recon-1", "em_name": "fibsem-uint16", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": True,
    },
    "jrc_hela-21": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "binary", "label_target": "nucleus", "axis_reverse": "none", "full_array": True,
    },
    "jrc_hela-22": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": True,
    },
    "jrc_hela-bfa": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "binary", "label_target": "nucleus", "axis_reverse": "none", "full_array": True,
    },
    "jrc_hela-h89-1": {
        "recon": "recon-1", "em_name": "fibsem-uint16", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": True,
    },
    "jrc_hela-h89-2": {
        "recon": "recon-1", "em_name": "fibsem-uint16", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": True,
    },
    "jrc_hela-nz-1": {
        "recon": "recon-2", "em_name": "fibsem-int16", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": True,
    },
    "jrc_hum-airway-14953vc": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nucleus", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": True,
    },
    # Stored in zarr v3 with sharding (1024^3 outer shards, 64^3 inner chunks); `_open_remote_zarr`
    # uses `FsspecStore` rather than `fsspec.get_mapper` so a crop only pulls the inner chunks it
    # actually needs, not whole ~1GB shard files. Its label pyramid has only one level (s0, full
    # resolution) - there is no genuinely coarse level to cheaply scan for a non-empty region, so
    # this uses a verified fixed crop instead of auto-detection.
    "jrc_mosquito-stylet-6": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": False,
        "bounding_box": ((1088, 1216), (640, 768), (768, 896)),
    },
    "jrc_mus-epididymis-1": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "binary", "label_target": "nucleus", "axis_reverse": "raw", "full_array": True,
    },
    "jrc_mus-epididymis-2": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "binary", "label_target": "nucleus", "axis_reverse": "raw", "full_array": True,
    },
    "jrc_mus-hippocampus-1": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": False,
    },
    # These three membranes are too thin/sparse to survive downsampling to the coarsest label
    # pyramid level, so auto-crop-detection finds no non-empty region; use a verified fixed crop.
    "jrc_mus-liver-4": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc_mem", "label_level": "s0",
        "label_kind": "binary", "label_target": "nuclear_membrane", "axis_reverse": "none", "full_array": False,
        "bounding_box": ((1409, 1537), (2000, 2128), (6000, 6128)),
    },
    "jrc_mus-liver-5": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc_mem", "label_level": "s0",
        "label_kind": "binary", "label_target": "nuclear_membrane", "axis_reverse": "none", "full_array": False,
        "bounding_box": ((2961, 3089), (5096, 5224), (4553, 4681)),
    },
    "jrc_mus-liver-6": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc_mem", "label_level": "s0",
        "label_kind": "binary", "label_target": "nuclear_membrane", "axis_reverse": "none", "full_array": False,
        "bounding_box": ((5600, 5792), (3970, 4162), (6610, 6802)),
    },
    "jrc_mus-pancreas-4": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": False,
    },
    "jrc_mus-sc-zp104a": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": False,
    },
    "jrc_mus-sc-zp105a": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": False,
    },
    "jrc_mus-skin-1": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": False,
    },
    "jrc_mus-thymus-1": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "instance", "label_target": "nucleus", "axis_reverse": "none", "full_array": True,
    },
    "aic_desmosome-3": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_subpath": "segmentations/nuc", "label_level": "s0",
        "label_kind": "binary", "label_target": "nucleus", "axis_reverse": "none", "full_array": True,
    },
}

# Datasets whose data license and paper could not be independently verified. Everything else in
# `DATASETS` is CC0-1.0 per the OpenOrganelle/CellMap project convention.
UNVERIFIED_LICENSE = ("jrc_mus-thymus-1", "aic_desmosome-3")

CROP_SHAPE_DEFAULT = (128, 128, 128)


def _open_remote_zarr(s3_path):
    import zarr
    from zarr.storage import FsspecStore

    # `FsspecStore` (not `fsspec.get_mapper`) is required for `jrc_mosquito-stylet-6`'s zarr v3
    # sharded array: it performs ranged partial reads of individual inner chunks, whereas
    # `fsspec.get_mapper` fetches whole shard files (up to ~1GB each) even for a small crop.
    store = FsspecStore.from_url(s3_path, storage_options={"anon": True}, read_only=True)
    return zarr.open(store, mode="r")


def _get_json(url):
    import requests

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _multiscale_levels(group_url):
    """Read multiscale pyramid levels from a zarr group, trying zarr v2 and v3 metadata.

    `jrc_mosquito-stylet-6`'s raw and label arrays are stored in zarr v3 (`zarr.json`), unlike
    every other dataset here (zarr v2, `.zattrs`); its label metadata nests multiscales under
    `attributes.ome`, while its raw metadata puts them directly under `attributes`.
    """
    try:
        multiscales = _get_json(f"{group_url}/.zattrs")["multiscales"]
    except Exception:
        attributes = _get_json(f"{group_url}/zarr.json")["attributes"]
        multiscales = attributes["ome"]["multiscales"] if "ome" in attributes else attributes["multiscales"]
    return {
        d["path"]: tuple(d["coordinateTransformations"][0]["scale"])
        for d in multiscales[0]["datasets"]
    }


def _find_matching_em_level(dataset_name, recon, em_name, label_scale):
    """Find the EM pyramid level whose scale exactly matches the label pyramid level, or `None`."""
    em_levels = _multiscale_levels(f"{BUCKET_URL}{dataset_name}/{dataset_name}.zarr/{recon}/em/{em_name}")
    for level, scale in em_levels.items():
        if scale == label_scale:
            return level, scale
    return None, None


def _nearest_em_level(dataset_name, recon, em_name, label_scale):
    """Find the EM pyramid level whose scale is closest to the label pyramid level.

    Used only for datasets whose label pyramid has no exact scale match.
    """
    em_levels = _multiscale_levels(f"{BUCKET_URL}{dataset_name}/{dataset_name}.zarr/{recon}/em/{em_name}")
    best_level, best_dist = None, None
    for level, scale in em_levels.items():
        dist = sum((s - lv) ** 2 for s, lv in zip(scale, label_scale))
        if best_dist is None or dist < best_dist:
            best_level, best_dist = level, dist
    return best_level, em_levels[best_level]


def _auto_bounding_box(dataset_name, recon, label_full_path, label_level, label_scale, crop_shape):
    """Center a crop on a non-empty region, found via the coarsest available label level.

    Downloading the coarsest level (rather than `label_level`, which may be a huge full-resolution
    array) keeps this cheap even for multi-TB datasets.
    """
    levels = _multiscale_levels(f"{BUCKET_URL}{dataset_name}/{dataset_name}.zarr/{recon}/{label_full_path}")
    coarsest_level = max(levels, key=lambda lv: levels[lv])
    coarse_url = (
        f"s3://janelia-cosem-datasets/{dataset_name}/{dataset_name}.zarr/{recon}/"
        f"{label_full_path}/{coarsest_level}"
    )
    coarse_arr = _open_remote_zarr(coarse_url)
    coarse_data = np.asarray(coarse_arr[:])
    nonzero = np.argwhere(coarse_data > 0)
    if len(nonzero) == 0:
        raise RuntimeError(f"No non-empty '{label_full_path}' region found for '{dataset_name}'.")
    center_coarse = nonzero[len(nonzero) // 2]
    scale_factor = np.array(levels[coarsest_level]) / np.array(label_scale)
    center = (center_coarse * scale_factor).astype(int)
    half = [c // 2 for c in crop_shape]
    return tuple((max(0, int(c - h)), int(c + h)) for c, h in zip(center, half))


def get_openorganelle_nucleus_data(path: Union[os.PathLike, str], dataset_name: str, download: bool = False) -> str:
    """Download nucleus segmentation data for one OpenOrganelle dataset.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        dataset_name: The name of the dataset, one of the keys in `DATASETS`.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the cached zarr store.
    """
    import zarr
    from zarr.codecs import BloscCodec
    from scipy.ndimage import zoom

    if dataset_name not in DATASETS:
        raise ValueError(f"'{dataset_name}' is not a valid dataset name. Choose from {sorted(DATASETS.keys())}.")

    info = DATASETS[dataset_name]
    recon, em_name = info["recon"], info["em_name"]
    label_subpath, label_level = info["label_subpath"], info["label_level"]
    axis_reverse, full_array = info["axis_reverse"], info["full_array"]

    os.makedirs(str(path), exist_ok=True)
    zarr_path = os.path.join(str(path), f"{dataset_name}.zarr")

    root = zarr.open_group(zarr_path, mode="a")
    if "raw" in root and "labels" in root:
        return zarr_path

    if not download:
        raise RuntimeError(f"No cached data found at '{zarr_path}'. Set download=True to stream it from S3.")

    print(f"Streaming nucleus data for '{dataset_name}' from the janelia-cosem-datasets S3 bucket ...")
    label_full_path = f"labels/inference/{label_subpath}"
    label_url = (
        f"s3://janelia-cosem-datasets/{dataset_name}/{dataset_name}.zarr/{recon}/{label_full_path}/{label_level}"
    )
    label_arr = _open_remote_zarr(label_url)

    label_scale = _multiscale_levels(
        f"{BUCKET_URL}{dataset_name}/{dataset_name}.zarr/{recon}/{label_full_path}"
    )[label_level]
    em_level, em_scale = _find_matching_em_level(dataset_name, recon, em_name, label_scale)
    resampled = em_level is None

    if full_array:
        label_data = np.asarray(label_arr[:])
        bounding_box = None
    else:
        bounding_box = _auto_bounding_box(
            dataset_name, recon, label_full_path, label_level, label_scale, CROP_SHAPE_DEFAULT,
        ) if info.get("bounding_box") is None else info["bounding_box"]
        label_slices = tuple(slice(*bb) for bb in bounding_box)
        label_data = label_arr[label_slices]

    if axis_reverse == "label":
        label_data = np.transpose(label_data, tuple(reversed(range(label_data.ndim))))
        raw_bbox = tuple(reversed(bounding_box)) if bounding_box is not None else None
    else:
        raw_bbox = bounding_box

    if em_level is not None:
        em_url = f"s3://janelia-cosem-datasets/{dataset_name}/{dataset_name}.zarr/{recon}/em/{em_name}/{em_level}"
        em_arr = _open_remote_zarr(em_url)
        if axis_reverse == "raw":
            raw_data = np.asarray(em_arr[:])
            raw_data = np.transpose(raw_data, tuple(reversed(range(raw_data.ndim))))
            if raw_bbox is not None:
                raw_data = raw_data[tuple(slice(*bb) for bb in raw_bbox)]
        elif raw_bbox is not None:
            raw_data = em_arr[tuple(slice(*bb) for bb in raw_bbox)]
        else:
            raw_data = np.asarray(em_arr[:])
    else:
        em_level, em_scale = _nearest_em_level(dataset_name, recon, em_name, label_scale)
        em_url = f"s3://janelia-cosem-datasets/{dataset_name}/{dataset_name}.zarr/{recon}/em/{em_name}/{em_level}"
        em_arr = _open_remote_zarr(em_url)
        ratio = [ls / es for ls, es in zip(label_scale, em_scale)]
        if raw_bbox is not None:
            em_bbox_native = tuple((int(lo * r), int(hi * r)) for (lo, hi), r in zip(raw_bbox, ratio))
            raw_native = em_arr[tuple(slice(*bb) for bb in em_bbox_native)]
        else:
            raw_native = np.asarray(em_arr[:])
        zoom_factors = [t / s for t, s in zip(label_data.shape, raw_native.shape)]
        raw_data = zoom(raw_native, zoom_factors, order=1).astype(raw_native.dtype)

    # Guard against off-by-a-few-voxel shape mismatches between independently stored pyramids.
    common_shape = tuple(min(r, lb) for r, lb in zip(raw_data.shape, label_data.shape))
    raw_data = raw_data[tuple(slice(0, s) for s in common_shape)]
    label_data = label_data[tuple(slice(0, s) for s in common_shape)]

    assert raw_data.shape == label_data.shape, (
        f"Shape mismatch for '{dataset_name}': raw {raw_data.shape} vs labels {label_data.shape}"
    )

    def _make_array(name, data, shuffle):
        arr = root.create_array(
            name, shape=data.shape, chunks=tuple(min(128, s) for s in data.shape), dtype=data.dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        arr[:] = data

    root.attrs["dataset"] = dataset_name
    root.attrs["recon"] = recon
    root.attrs["label_kind"] = info["label_kind"]
    root.attrs["label_target"] = info["label_target"]
    root.attrs["label_source"] = label_url
    root.attrs["em_source"] = em_url
    root.attrs["resampled"] = resampled
    root.attrs["license_verified"] = dataset_name not in UNVERIFIED_LICENSE

    _make_array("raw", raw_data, shuffle="shuffle")
    _make_array("labels", label_data, shuffle="bitshuffle")

    print(f"Cached '{dataset_name}' to '{zarr_path}' (shape {raw_data.shape}).")
    return zarr_path


def get_openorganelle_nucleus_paths(
    path: Union[os.PathLike, str], dataset_names: Optional[List[str]] = None, download: bool = False,
) -> List[str]:
    """Get paths to cached OpenOrganelle nucleus zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        dataset_names: The names of the datasets to use. Defaults to all datasets in `DATASETS`.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    if dataset_names is None:
        dataset_names = list(DATASETS.keys())
    return [get_openorganelle_nucleus_data(path, name, download) for name in dataset_names]


def get_openorganelle_nucleus_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    dataset_names: Optional[List[str]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the OpenOrganelle nucleus dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        dataset_names: The names of the datasets to use. Defaults to all datasets in `DATASETS`.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_openorganelle_nucleus_paths(path, dataset_names, download)

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


def get_openorganelle_nucleus_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    dataset_names: Optional[List[str]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for nucleus segmentation in the OpenOrganelle nucleus dataset.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        dataset_names: The names of the datasets to use. Defaults to all datasets in `DATASETS`.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_openorganelle_nucleus_dataset(
        path, patch_shape, dataset_names=dataset_names, download=download,
        offsets=offsets, boundaries=boundaries, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
