"""This dataset provides nucleus segmentation masks for representative crops from three
"OpenOrganelle" volumes (the public `janelia-cosem-datasets` S3 bucket, the same bucket used
by `cellmap.py` and `janelia_nucleus.py`), not covered by either of those existing loaders:

- jrc_ctl-id8-2: a binary nucleus mask, sharing an identical multiscale pyramid with its FIB-SEM
  raw data (no resampling or axis reordering needed).
- jrc_choroid-plexus-2: nucleus instance labels, stored with a reversed axis order relative to its
  FIB-SEM raw data (corrected internally by this module).
- jrc_dauer-larva: nucleus instance labels for a TEM volume, whose label pyramid has no exact-scale
  match in the raw pyramid; the raw crop is resampled onto the label grid.

The label data is released at `{dataset}.zarr/{recon}/labels/inference/nucleus_seg/{level}` on the
bucket, a third naming convention distinct from the one `janelia_nucleus.py` uses. All 3 volumes are
released under CC0-1.0. The full matched-resolution raw+label arrays are far too large to download
whole (tens of GB to ~1.8 TB); this module instead downloads one fixed, manually verified crop per
dataset, matching the crop that was visually reviewed in napari before this loader was added.

Please cite the CellMap project (https://www.janelia.org/project-team/cellmap) if you use this data.
"""

import os
from typing import List, Optional, Tuple, Union

import numpy as np

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


BUCKET_URL = "https://janelia-cosem-datasets.s3.amazonaws.com/"
LABEL_SUBPATH = "labels/inference/nucleus_seg"

# dataset name -> recon folder, em array name, nucleus_seg pyramid level, label kind, whether the
# label array's axis order must be reversed to align with the raw data, and the crop (in label-level
# voxel coordinates) that was visually reviewed and approved before this loader was added.
# "binary" = label array only ever takes values {0, 1} (semantic nucleus mask).
# "instance" = label array carries per-nucleus integer instance IDs.
DATASETS = {
    "jrc_ctl-id8-2": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_level": "s0", "label_kind": "binary",
        "needs_axis_reverse": False, "bounding_box": ((3824, 4048), (448, 672), (4064, 4288)),
    },
    "jrc_choroid-plexus-2": {
        "recon": "recon-1", "em_name": "fibsem-uint8", "label_level": "s1", "label_kind": "instance",
        "needs_axis_reverse": True, "bounding_box": ((780, 1100), (2220, 2540), (620, 940)),
    },
    "jrc_dauer-larva": {
        "recon": "recon-1", "em_name": "tem-uint8", "label_level": "s0", "label_kind": "instance",
        "needs_axis_reverse": False, "bounding_box": ((130, 330), (1804, 2004), (11724, 11924)),
    },
}


def _open_remote_zarr(s3_path):
    import zarr
    import fsspec

    store = fsspec.get_mapper(s3_path, anon=True)
    return zarr.open(store, mode="r")


def _get_json(url):
    import requests

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _multiscale_levels(zattrs_url):
    attrs = _get_json(zattrs_url)
    return {
        d["path"]: tuple(d["coordinateTransformations"][0]["scale"])
        for d in attrs["multiscales"][0]["datasets"]
    }


def _find_matching_em_level(dataset_name, recon, em_name, label_scale):
    """Find the EM pyramid level whose scale exactly matches the label pyramid level, or `None`."""
    em_levels = _multiscale_levels(f"{BUCKET_URL}{dataset_name}/{dataset_name}.zarr/{recon}/em/{em_name}/.zattrs")
    for level, scale in em_levels.items():
        if scale == label_scale:
            return level, scale
    return None, None


def _nearest_em_level(dataset_name, recon, em_name, label_scale):
    """Find the EM pyramid level whose scale is closest to the label pyramid level.

    Used only for `jrc_dauer-larva`, whose label pyramid has no exact scale match.
    """
    em_levels = _multiscale_levels(f"{BUCKET_URL}{dataset_name}/{dataset_name}.zarr/{recon}/em/{em_name}/.zattrs")
    best_level, best_dist = None, None
    for level, scale in em_levels.items():
        dist = sum((s - lv) ** 2 for s, lv in zip(scale, label_scale))
        if best_dist is None or dist < best_dist:
            best_level, best_dist = level, dist
    return best_level, em_levels[best_level]


def get_openorganelle_nucleus_data(path: Union[os.PathLike, str], dataset_name: str, download: bool = False) -> str:
    """Download the reviewed nucleus segmentation crop for one OpenOrganelle dataset.

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
    recon, em_name, label_level = info["recon"], info["em_name"], info["label_level"]
    needs_axis_reverse, bounding_box = info["needs_axis_reverse"], info["bounding_box"]

    os.makedirs(str(path), exist_ok=True)
    zarr_path = os.path.join(str(path), f"{dataset_name}.zarr")

    root = zarr.open_group(zarr_path, mode="a")
    if "raw" in root and "labels" in root:
        return zarr_path

    if not download:
        raise RuntimeError(f"No cached crop found at '{zarr_path}'. Set download=True to stream it from S3.")

    print(f"Streaming a nucleus crop for '{dataset_name}' from the janelia-cosem-datasets S3 bucket ...")
    label_url = f"s3://janelia-cosem-datasets/{dataset_name}/{dataset_name}.zarr/{recon}/{LABEL_SUBPATH}/{label_level}"
    label_arr = _open_remote_zarr(label_url)
    label_slices = tuple(slice(*bb) for bb in bounding_box)
    label_crop = label_arr[label_slices]

    if needs_axis_reverse:
        label_crop_zyx = np.transpose(label_crop, tuple(reversed(range(label_crop.ndim))))
        em_bbox = tuple(reversed(bounding_box))
    else:
        label_crop_zyx = label_crop
        em_bbox = bounding_box

    label_scale = _multiscale_levels(
        f"{BUCKET_URL}{dataset_name}/{dataset_name}.zarr/{recon}/{LABEL_SUBPATH}/.zattrs"
    )[label_level]

    em_level, em_scale = _find_matching_em_level(dataset_name, recon, em_name, label_scale)
    resampled = em_level is None
    if em_level is not None:
        em_url = f"s3://janelia-cosem-datasets/{dataset_name}/{dataset_name}.zarr/{recon}/em/{em_name}/{em_level}"
        em_arr = _open_remote_zarr(em_url)
        raw_crop = em_arr[tuple(slice(*bb) for bb in em_bbox)]
    else:
        em_level, em_scale = _nearest_em_level(dataset_name, recon, em_name, label_scale)
        em_url = f"s3://janelia-cosem-datasets/{dataset_name}/{dataset_name}.zarr/{recon}/em/{em_name}/{em_level}"
        em_arr = _open_remote_zarr(em_url)
        ratio = [ls / es for ls, es in zip(label_scale, em_scale)]
        em_bbox_native = tuple((int(lo * r), int(hi * r)) for (lo, hi), r in zip(em_bbox, ratio))
        raw_native = em_arr[tuple(slice(*bb) for bb in em_bbox_native)]
        zoom_factors = [t / s for t, s in zip(label_crop_zyx.shape, raw_native.shape)]
        raw_crop = zoom(raw_native, zoom_factors, order=1).astype(raw_native.dtype)

    assert raw_crop.shape == label_crop_zyx.shape, (
        f"Shape mismatch for '{dataset_name}': raw {raw_crop.shape} vs labels {label_crop_zyx.shape}"
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
    root.attrs["label_source"] = label_url
    root.attrs["em_source"] = em_url
    root.attrs["bounding_box"] = bounding_box
    root.attrs["resampled"] = resampled

    _make_array("raw", raw_crop, shuffle="shuffle")
    _make_array("labels", label_crop_zyx, shuffle="bitshuffle")

    print(f"Cached '{dataset_name}' to '{zarr_path}' (shape {raw_crop.shape}).")
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
