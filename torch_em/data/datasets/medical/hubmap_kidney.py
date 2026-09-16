"""The HuBMAP - Hacking the Kidney dataset contains annotations for glomeruli functional tissue unit (FTU)
segmentation in human kidney whole-slide histopathology images (PAS-stained).

This is the dataset for the "HuBMAP - Hacking the Kidney" Kaggle competition, located at
https://www.kaggle.com/competitions/hubmap-kidney-segmentation. It provides 8 PAS-stained kidney whole-slide
images (fresh-frozen and FFPE tissue preparations, contributed by the Human BioMolecular Atlas Program /
HuBMAP through the BIOMIC team at Vanderbilt University), each paired with expert-annotated glomerulus
segmentations, given as run-length encoded (RLE) masks in 'train.csv'. The competition data is released
under the CC BY 4.0 license.

NOTE: This is a DIFFERENT Kaggle competition from "HuBMAP + HPA - Hacking the Human Body" (see
'hubmap_hpa.py' for that dataset), even though both are hosted by HuBMAP and target FTU segmentation.

NOTE: Downloading this dataset requires a Kaggle account that has accepted the competition rules at
https://www.kaggle.com/competitions/hubmap-kidney-segmentation/rules. Without that, the Kaggle API
download fails with an HTTP 403 error, even with valid API credentials.

NOTE: The whole-slide images are very large single-resolution BigTIFFs (tens of thousands of pixels per
side, several hundred MB to multiple GB per file). As in 'histopathology/camelyon.py', each image is read
lazily (via a 'zarr' view over the tiled TIFF where available, otherwise a memory-mapped read) and converted
once into a chunked HDF5 file that stores the raw image together with the binary glomerulus mask decoded
from the RLE encoding. Downstream patch extraction for training then happens lazily via
'torch_em.default_segmentation_dataset' from these HDF5 files, so no manual tiling logic is required here.

This dataset is described in the publication https://doi.org/10.1038/s41467-023-40291-0 ("Segmenting
functional tissue units across human organs using community-driven development of generalizable machine
learning algorithms", Nature Communications, 2023). Please cite it if you use this dataset in your research.
"""

import os
import csv
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util

# The RLE-encoded masks in 'train.csv' exceed the default field size limit for these large WSIs.
csv.field_size_limit(10**8)


def _rle_decode(rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """Decode a Kaggle-style run-length-encoded mask into a binary array.

    The encoding lists 1-indexed (start, length) pairs over a column-major (Fortran order) flattening of
    the image, i.e. pixels are numbered from top to bottom, then left to right.
    """
    height, width = shape
    mask = np.zeros(height * width, dtype=np.uint8)

    values = [int(v) for v in rle.split()]
    starts, lengths = values[0::2], values[1::2]
    for start, length in zip(starts, lengths):
        start -= 1
        mask[start:start + length] = 1

    return mask.reshape((width, height)).T


def _open_raw_array(tiff_path):
    import tifffile

    tiff = tifffile.TiffFile(tiff_path)
    # Some slides are stored as several series (e.g. a full-resolution image plus a thumbnail);
    # pick the one with the most pixels rather than assuming series[0] is the full-resolution one.
    series = max(tiff.series, key=lambda s: np.prod(s.shape))

    try:
        import zarr
        array = zarr.open(series.aszarr(), mode="r")
        if not hasattr(array, "shape"):  # The tiff is pyramidal: pick the largest resolution level.
            array = max(array.values(), key=lambda level: np.prod(level.shape))
    except Exception:
        array = series.asarray()

    # Some slides carry extra singleton axes (e.g. an OME-style (T, Z, C, H, W) layout), which have
    # to be dropped before the channel axis can be identified. Indexed rather than via '.squeeze()',
    # so a lazy zarr array is not fully materialized just to drop size-1 axes.
    squeeze_axes = tuple(i for i, size in enumerate(array.shape) if size == 1)
    if squeeze_axes:
        index = tuple(0 if i in squeeze_axes else slice(None) for i in range(array.ndim))
        array = array[index]

    # Some HuBMAP kidney tiffs store the channel axis first, i.e. (3, height, width) instead of the usual
    # channel-last (height, width, 3) layout. Normalize to channel-last so downstream code is uniform.
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))

    assert array.ndim == 3 and array.shape[-1] == 3, f"Unexpected array shape {array.shape} for {tiff_path}."
    return array


def _convert_slide(tiff_path: str, rle: str, output_path: str, tile: int = 4096) -> None:
    import h5py

    array = _open_raw_array(tiff_path)
    height, width = array.shape[0], array.shape[1]
    mask = _rle_decode(rle, (height, width))

    tmp_path = output_path + ".tmp"
    with h5py.File(tmp_path, "w") as f:
        raw = f.create_dataset(
            "raw", shape=(3, height, width), dtype="uint8", compression="gzip", chunks=(1, 512, 512)
        )
        labels = f.create_dataset(
            "labels", shape=(height, width), dtype="uint8", compression="gzip", chunks=(512, 512)
        )
        for y in tqdm(range(0, height, tile), desc=f"Converting {Path(tiff_path).stem}"):
            for x in range(0, width, tile):
                th, tw = min(tile, height - y), min(tile, width - x)
                tile_data = np.asarray(array[y:y + th, x:x + tw])
                raw[:, y:y + th, x:x + tw] = tile_data.transpose(2, 0, 1)
                labels[y:y + th, x:x + tw] = mask[y:y + th, x:x + tw]

    os.replace(tmp_path, output_path)


def _load_train_annotations(data_dir):
    csv_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(csv_path):
        raise RuntimeError(f"Could not find 'train.csv' at {csv_path}.")
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))


def get_hubmap_kidney_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the HuBMAP - Hacking the Kidney dataset.

    Args:
        path: Filepath to a folder where the data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the raw data is stored.
    """
    data_dir = os.path.join(path, "train")
    if os.path.exists(data_dir):
        return path

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "hubmap-kidney-segmentation.zip")
    util.download_source_kaggle(
        path=path, dataset_name="hubmap-kidney-segmentation", download=download, competition=True
    )
    util.unzip(zip_path=zip_path, dst=path)

    if not os.path.exists(data_dir):
        raise RuntimeError(f"The expected 'train' folder is missing after extraction at {path}.")
    return path


def get_hubmap_kidney_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the preprocessed HuBMAP - Hacking the Kidney data.

    Each returned HDF5 file stores the raw image under 'raw' (channels, height, width) and the
    binary glomerulus segmentation mask under 'labels' (height, width).

    Args:
        path: Filepath to a folder where the data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the preprocessed HDF5 files.
    """
    data_dir = get_hubmap_kidney_data(path, download)
    rows = _load_train_annotations(data_dir)

    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    volume_paths = []
    for row in rows:
        image_id = row["id"]
        tiff_path = os.path.join(data_dir, "train", f"{image_id}.tiff")
        if not os.path.exists(tiff_path):
            continue

        output_path = os.path.join(preprocessed_dir, f"{image_id}.h5")
        if not os.path.exists(output_path):
            _convert_slide(tiff_path, row["encoding"], output_path)
        volume_paths.append(output_path)

    if not volume_paths:
        raise RuntimeError(f"No annotated images were found at {data_dir}.")

    return volume_paths


def get_hubmap_kidney_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    download: bool = False,
    resize_inputs: bool = False,
    **kwargs,
) -> Dataset:
    """Get the HuBMAP - Hacking the Kidney dataset for glomerulus segmentation in kidney whole-slide images.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        resize_inputs: Whether to resize the inputs.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_hubmap_kidney_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        with_channels=True,
        ndim=2,
        **kwargs,
    )


def get_hubmap_kidney_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    download: bool = False,
    resize_inputs: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the HuBMAP - Hacking the Kidney dataloader for glomerulus segmentation in kidney whole-slide images.

    Args:
        path: Filepath to a folder where the data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        resize_inputs: Whether to resize the inputs.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hubmap_kidney_dataset(path, patch_shape, download, resize_inputs, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
