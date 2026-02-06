"""The XPRESS dataset contains volumetric microscopy data with voxel-wise labels.

The training data is hosted at:
- https://github.com/htem/xpress-challenge-files/releases/download/v1.0/xpress-training-raw.h5
- https://github.com/htem/xpress-challenge-files/releases/download/v1.0/xpress-training-voxel-labels.h5
"""

import os
from typing import Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "raw": "https://github.com/htem/xpress-challenge-files/releases/download/v1.0/xpress-training-raw.h5",
    "labels": "https://github.com/htem/xpress-challenge-files/releases/download/v1.0/xpress-training-voxel-labels.h5",
}


def _default_chunks(shape):
    # Simple heuristic: chunk along z and limit chunk extents to 64.
    return tuple(min(64, int(s)) for s in shape)


def _copy_chunked(src, dst, chunks):
    shape = src.shape
    if len(shape) < 3:
        raise ValueError(f"Expected at least 3D data, got shape={shape}.")

    z_chunk = chunks[0]
    for z in range(0, shape[0], z_chunk):
        z_slice = slice(z, min(z + z_chunk, shape[0]))
        dst[(z_slice, *[slice(None)] * (len(shape) - 1))] = src[(z_slice, *[slice(None)] * (len(shape) - 1))]


def _merge_to_single_h5(raw_path: Union[os.PathLike, str], label_path: Union[os.PathLike, str], out_path: str):
    if os.path.exists(out_path):
        return out_path

    import h5py
    from skimage.transform import resize

    with h5py.File(raw_path, "r") as fr, h5py.File(label_path, "r") as fl, h5py.File(out_path, "w") as fo:
        raw = fr["volumes/raw"]
        labels = fl["volumes/labels"]

        raw_chunks = raw.chunks if raw.chunks is not None else _default_chunks(raw.shape)

        if labels.shape != raw.shape:
            # Nearest-neighbor upsampling for labels to match raw shape.
            labels_arr = resize(
                labels[...],
                raw.shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(labels.dtype, copy=False)
            label_shape = labels_arr.shape
        else:
            labels_arr = None
            label_shape = labels.shape

        label_chunks = labels.chunks if labels.chunks is not None else _default_chunks(label_shape)

        raw_ds = fo.create_dataset(
            "raw",
            shape=raw.shape,
            dtype=raw.dtype,
            chunks=raw_chunks,
            compression="gzip",
            compression_opts=4,
        )
        label_ds = fo.create_dataset(
            "labels",
            shape=label_shape,
            dtype=labels.dtype,
            chunks=label_chunks,
            compression="gzip",
            compression_opts=4,
        )

        _copy_chunked(raw, raw_ds, raw_chunks)
        if labels_arr is None:
            _copy_chunked(labels, label_ds, label_chunks)
        else:
            label_ds[...] = labels_arr

    return out_path


def get_xpress_data(path: Union[os.PathLike, str], download: bool = False) -> Tuple[str, str]:
    """Download the XPRESS training data.

    Args:
        path: Filepath to a folder where the data will be stored.
        download: Whether to download the data if it is not present.

    Returns:
        Filepaths for raw and label data.
    """
    os.makedirs(path, exist_ok=True)
    raw_path = os.path.join(path, "xpress-training-raw.h5")
    label_path = os.path.join(path, "xpress-training-voxel-labels.h5")

    util.download_source(raw_path, URLS["raw"], download, checksum=None)
    util.download_source(label_path, URLS["labels"], download, checksum=None)

    merged_path = os.path.join(path, "xpress-training.h5")
    _merge_to_single_h5(raw_path, label_path, merged_path)

    return merged_path, merged_path


def get_xpress_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[str, str]:
    """Get paths to the XPRESS training data."""
    return get_xpress_data(path, download)


def get_xpress_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    raw_key: Optional[str] = None,
    label_key: Optional[str] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the XPRESS dataset for voxel-wise segmentation.

    Args:
        path: Filepath to a folder where the data will be stored.
        patch_shape: The patch shape to use for training.
        raw_key: The HDF5 key for the raw data. If None, it will be inferred when possible.
        label_key: The HDF5 key for the label data. If None, it will be inferred when possible.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3
    raw_path, label_path = get_xpress_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=[raw_path],
        raw_key="raw" if raw_key is None else raw_key,
        label_paths=[label_path],
        label_key="labels" if label_key is None else label_key,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs,
    )


def get_xpress_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    raw_key: Optional[str] = None,
    label_key: Optional[str] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the XPRESS dataloader for voxel-wise segmentation."""
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_xpress_dataset(
        path, patch_shape, raw_key=raw_key, label_key=label_key, download=download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
