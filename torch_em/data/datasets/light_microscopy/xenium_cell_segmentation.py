"""Xenium output bundles contain morphology images and cell/nucleus segmentation masks.

This loader supports local Xenium Onboard Analysis output directories. It reads the true segmentation masks from
`cells.zarr.zip` or `cells.zarr` (`masks/0` for nuclei and `masks/1` for cells), stores them together with all
selected morphology channels in HDF5, and exposes the data through the usual torch-em segmentation dataset API.

The output format is described in the 10x Genomics documentation:
https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/advanced/xoa-output-zarr
"""

import os
from glob import glob
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


RAW_CHANNEL_NAMES = {
    0: "dapi",
    1: "boundary",
    2: "interior_rna",
    3: "interior_protein",
}


def _open_zarr(path):
    import zarr

    if path.endswith(".zip"):
        try:
            from zarr.storage import ZipStore
        except ImportError:
            store = zarr.ZipStore(path, mode="r")
        else:
            store = ZipStore(path, mode="r")
        return zarr.open_group(store=store, mode="r")

    try:
        from zarr.storage import LocalStore
    except ImportError:
        store = zarr.DirectoryStore(path)
        return zarr.open_group(store=store, mode="r")

    store = LocalStore(path)
    return zarr.open_group(store=store, mode="r")


def _read_mask(cells_zarr, index):
    root = _open_zarr(cells_zarr)
    return np.asarray(root["masks"][str(index)])


def _read_tiff(path, series=0, level=0):
    import tifffile

    with tifffile.TiffFile(path) as f:
        tiff_series = f.series[series]
        if hasattr(tiff_series, "levels") and len(tiff_series.levels) > level:
            return tiff_series.levels[level].asarray()
        return tiff_series.asarray()


def _to_channel_stack(image, projection, z_index):
    image = np.squeeze(image)
    if image.ndim == 2:
        return image[None]

    if image.ndim == 3:
        if projection == "max":
            return image.max(axis=0, keepdims=True)
        if projection == "mean":
            return image.mean(axis=0, keepdims=True).astype(image.dtype, copy=False)
        if projection == "slice":
            return image[z_index][None]

    spatial_shape = image.shape[-2:]
    return image.reshape((-1,) + spatial_shape)


def _find_cells_zarr(sample_dir):
    candidates = [
        os.path.join(sample_dir, "cells.zarr.zip"),
        os.path.join(sample_dir, "cells.zarr"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _find_morphology_files(sample_dir):
    focus_dir = os.path.join(sample_dir, "morphology_focus")
    if os.path.isdir(focus_dir):
        ome_tifs = set(glob(os.path.join(focus_dir, "*.ome.tif")))
        tifs = set(glob(os.path.join(focus_dir, "*.tif")))
        focus_files = sorted(ome_tifs | tifs)
        if focus_files:
            return focus_files

    candidates = [
        "morphology_focus.ome.tif",
        "morphology_mip.ome.tif",
        "morphology.ome.tif",
    ]
    for candidate in candidates:
        path = os.path.join(sample_dir, candidate)
        if os.path.exists(path):
            return [path]

    return []


def _discover_sample_dirs(path, sample_ids):
    if sample_ids is not None:
        if isinstance(sample_ids, str):
            sample_ids = [sample_ids]
        sample_dirs = [os.path.join(path, sample_id) for sample_id in sample_ids]
    elif _find_cells_zarr(path) is not None:
        sample_dirs = [path]
    else:
        sample_dirs = [p for p in sorted(glob(os.path.join(path, "*"))) if os.path.isdir(p) and _find_cells_zarr(p)]

    missing = [sample_dir for sample_dir in sample_dirs if _find_cells_zarr(sample_dir) is None]
    if missing:
        raise RuntimeError(
            "Could not find `cells.zarr.zip` or `cells.zarr` for the following Xenium samples: "
            f"{missing}."
        )
    return sample_dirs


def _normalize_channel_selection(raw_channels, channel_names):
    if raw_channels is None:
        return list(range(len(channel_names)))
    if isinstance(raw_channels, (str, int)):
        raw_channels = [raw_channels]

    selected = []
    for channel in raw_channels:
        if isinstance(channel, int):
            if channel < 0 or channel >= len(channel_names):
                raise ValueError(
                    f"Invalid raw channel index: {channel}. Available indices are 0 to {len(channel_names) - 1}."
                )
            selected.append(channel)
        else:
            if channel not in channel_names:
                raise ValueError(f"Unknown raw channel '{channel}'. Available channels are {channel_names}.")
            selected.append(channel_names.index(channel))
    return selected


def _read_morphology_channels(sample_dir, projection, z_index):
    morphology_files = _find_morphology_files(sample_dir)
    if not morphology_files:
        raise RuntimeError(f"Could not find Xenium morphology images in {sample_dir}.")

    channels, names = [], []
    if len(morphology_files) > 1:
        for channel_id, morphology_file in enumerate(morphology_files):
            channel = _to_channel_stack(_read_tiff(morphology_file), projection="slice", z_index=z_index)
            if len(channel) != 1:
                channel = channel[:1]
            channels.append(channel[0])
            names.append(RAW_CHANNEL_NAMES.get(channel_id, f"channel_{channel_id}"))
    else:
        channel_stack = _to_channel_stack(_read_tiff(morphology_files[0]), projection=projection, z_index=z_index)
        channels = list(channel_stack)
        names = [RAW_CHANNEL_NAMES.get(channel_id, f"channel_{channel_id}") for channel_id in range(len(channels))]

    return channels, names


def _preprocess_sample(sample_dir, output_path, raw_channels, projection, z_index):
    import h5py

    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            if raw_channels is None and bool(f.attrs.get("all_channels_stored", False)):
                return
            if raw_channels is not None and "raw" in f:
                requested = raw_channels if isinstance(raw_channels, (list, tuple)) else [raw_channels]
                raw_keys = set(f["raw"].keys())
                if all((f"channel_{ch}" if isinstance(ch, int) else ch) in raw_keys for ch in requested):
                    return
        os.remove(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    channels, channel_names = _read_morphology_channels(sample_dir, projection=projection, z_index=z_index)
    selected_channels = _normalize_channel_selection(raw_channels, channel_names)

    raw = np.stack([channels[channel_id] for channel_id in selected_channels])
    selected_names = [channel_names[channel_id] for channel_id in selected_channels]

    cells_zarr = _find_cells_zarr(sample_dir)
    nuclei = _read_mask(cells_zarr, 0)
    cells = _read_mask(cells_zarr, 1)

    if raw.shape[1:] != nuclei.shape:
        raise RuntimeError(
            f"Shape mismatch for {sample_dir}: raw has shape {raw.shape[1:]}, "
            f"but nucleus labels have shape {nuclei.shape}."
        )

    with h5py.File(output_path, "w") as f:
        f.attrs["sample_dir"] = sample_dir
        f.attrs["raw_channel_names"] = np.asarray(selected_names, dtype="S")
        f.attrs["all_channels_stored"] = raw_channels is None
        f.create_dataset("raw/all", data=raw, compression="gzip")
        for channel_id, (channel, name) in enumerate(zip(raw, selected_names)):
            f.create_dataset(f"raw/{name}", data=channel, compression="gzip")
            if name != f"channel_{channel_id}":
                f.create_dataset(f"raw/channel_{channel_id}", data=channel, compression="gzip")
        f.create_dataset("labels/nucleus", data=nuclei, compression="gzip")
        f.create_dataset("labels/cell", data=cells, compression="gzip")


def get_xenium_cell_segmentation_data(
    path: Union[os.PathLike, str],
    sample_ids: Optional[Union[str, Sequence[str]]] = None,
    raw_channels: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
    projection: Literal["max", "mean", "slice"] = "max",
    z_index: int = 0,
    download: bool = False,
) -> str:
    """Preprocess local Xenium output bundles into HDF5 files.

    Args:
        path: Filepath to a Xenium output directory or to a folder with one or more output directories.
        sample_ids: Optional sample directory names if `path` contains multiple Xenium outputs.
        raw_channels: The morphology channels to store. By default all discovered channels are stored.
        projection: How to convert a 3D morphology image into a 2D image, if needed.
        z_index: The z-slice used when `projection='slice'`.
        download: Placeholder for API consistency. Automatic download is not supported for Xenium.

    Returns:
        The folder with preprocessed HDF5 files.
    """
    if download:
        raise RuntimeError(
            "Automatic download is not supported for Xenium because 10x publishes multiple large example datasets. "
            "Please download the desired Xenium output bundle from 10x Genomics and pass its local path."
        )
    if projection not in ("max", "mean", "slice"):
        raise ValueError(f"Invalid projection '{projection}'. Choose from 'max', 'mean' or 'slice'.")

    sample_dirs = _discover_sample_dirs(path, sample_ids)
    output_root = os.path.join(path, "preprocessed")
    for sample_dir in sample_dirs:
        sample_name = os.path.basename(os.path.normpath(sample_dir))
        output_path = os.path.join(output_root, f"{sample_name}.h5")
        _preprocess_sample(sample_dir, output_path, raw_channels=raw_channels, projection=projection, z_index=z_index)

    return output_root


def get_xenium_cell_segmentation_paths(
    path: Union[os.PathLike, str],
    sample_ids: Optional[Union[str, Sequence[str]]] = None,
    raw_channels: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
    projection: Literal["max", "mean", "slice"] = "max",
    z_index: int = 0,
    download: bool = False,
) -> List[str]:
    """Get paths to preprocessed Xenium HDF5 files."""
    output_root = get_xenium_cell_segmentation_data(path, sample_ids, raw_channels, projection, z_index, download)
    paths = sorted(glob(os.path.join(output_root, "*.h5")))
    if not paths:
        raise RuntimeError(f"Could not find any preprocessed Xenium files in {output_root}.")
    return paths


def get_xenium_cell_segmentation_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    label_kind: Literal["nucleus", "cell"] = "cell",
    raw_channel: Union[str, int] = "all",
    sample_ids: Optional[Union[str, Sequence[str]]] = None,
    raw_channels: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
    projection: Literal["max", "mean", "slice"] = "max",
    z_index: int = 0,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Xenium dataset for nucleus or cell segmentation.

    Args:
        path: Filepath to a Xenium output directory or to a folder with one or more output directories.
        patch_shape: The patch shape to use for training.
        label_kind: Which segmentation labels to load. Either "nucleus" or "cell".
        raw_channel: Which raw data to load from the preprocessed file. Use "all" for the full channel stack,
            a channel name such as "dapi", or a channel index.
        sample_ids: Optional sample directory names if `path` contains multiple Xenium outputs.
        raw_channels: The morphology channels to store. By default all discovered channels are stored.
        projection: How to convert a 3D morphology image into a 2D image, if needed.
        z_index: The z-slice used when `projection='slice'`.
        download: Placeholder for API consistency. Automatic download is not supported for Xenium.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if label_kind not in ("nucleus", "cell"):
        raise ValueError(f"Invalid label kind '{label_kind}'. Choose from 'nucleus' or 'cell'.")

    paths = get_xenium_cell_segmentation_paths(path, sample_ids, raw_channels, projection, z_index, download)
    if isinstance(raw_channel, int):
        raw_key = f"raw/channel_{raw_channel}"
        with_channels = False
    elif raw_channel == "all":
        raw_key = "raw/all"
        with_channels = True
    else:
        raw_key = f"raw/{raw_channel}"
        with_channels = False

    kwargs = util.update_kwargs(kwargs, "with_channels", with_channels)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key=raw_key,
        label_paths=paths,
        label_key=f"labels/{label_kind}",
        patch_shape=patch_shape,
        ndim=2,
        **kwargs
    )


def get_xenium_cell_segmentation_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    label_kind: Literal["nucleus", "cell"] = "cell",
    raw_channel: Union[str, int] = "all",
    sample_ids: Optional[Union[str, Sequence[str]]] = None,
    raw_channels: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
    projection: Literal["max", "mean", "slice"] = "max",
    z_index: int = 0,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Xenium dataloader for nucleus or cell segmentation."""
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_xenium_cell_segmentation_dataset(
        path, patch_shape, label_kind=label_kind, raw_channel=raw_channel, sample_ids=sample_ids,
        raw_channels=raw_channels, projection=projection, z_index=z_index, download=download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
