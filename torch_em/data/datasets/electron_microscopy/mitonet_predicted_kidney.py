"""The MitoNet-Predicted-Kidney dataset streams mitochondria instance segmentation for the Janelia
OpenOrganelle mouse kidney FIB-SEM volume (jrc_mus-kidney) at 16 nm isotropic resolution.

IMPORTANT: The labels are automatically generated instance segmentation predictions from MitoNet
(Conrad and Narayan 2022), not manually annotated ground truth. They are hardened at a semantic
confidence threshold of 0.5 and a contour confidence threshold of 0.3, followed by connected components
and watershed. Use them as weak or pseudo-labels, not as verified ground truth.

Raw data is streamed from the public OpenOrganelle S3 bucket. Predicted labels are streamed from a
remote Zarr array embedded in a Zenodo/figshare ZIP archive, decoded chunk-by-chunk via HTTP range
requests. Only the requested bounding box is downloaded and cached locally as a zarr v3 store.

The predictions are available at https://doi.org/10.6084/m9.figshare.20749729, licensed CC-BY-4.0.
The MitoNet method is described in https://doi.org/10.1016/j.cels.2022.12.004.
Please cite this publication if you use the predictions in your research.
"""

import os
from typing import List, Tuple, Union

import numpy as np

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


RAW_S3_URL = "s3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.zarr/recon-1/em/fibsem-uint8/s1"
LABEL_ZIP_URL = "https://ndownloader.figshare.com/files/36985378"
LABEL_ARRAY_PATH = "kidney16nm.zarr/empanada_mito_pred"

FULL_SHAPE = (11099, 3988, 6143)  # (z, y, x) at 16 nm isotropic resolution.
LABEL_CHUNK_SHAPE = (512, 512, 512)


def _bbox_hash(bounding_box):
    import hashlib
    return hashlib.md5("_".join(str(v) for v in bounding_box).encode()).hexdigest()[:12]


def _read_raw_block(z_min, z_max, y_min, y_max, x_min, x_max):
    """Slice the requested block directly from the public OpenOrganelle S3 zarr array."""
    import zarr
    import fsspec

    store = fsspec.get_mapper(RAW_S3_URL, anon=True)
    raw = zarr.open(store, mode="r")
    return np.asarray(raw[z_min:z_max, y_min:y_max, x_min:x_max])


def _read_label_block(z_min, z_max, y_min, y_max, x_min, x_max):
    """Decode only the chunks overlapping the requested block from the remote ZIP-embedded zarr array.

    `zarr.open` cannot be used directly here: the archive's zarr v2 store has no `.zattrs` entry, and
    `fsspec`'s zip backend raises instead of treating that as optional. The array layout (shape, chunk
    shape, dtype, blosc codec) is fixed and taken from the published `.zarray` metadata instead.
    """
    import zipfile
    import fsspec
    import numcodecs

    fs = fsspec.filesystem("http")
    zf = zipfile.ZipFile(fs.open(LABEL_ZIP_URL, "rb"))
    codec = numcodecs.Blosc()

    cz, cy, cx = LABEL_CHUNK_SHAPE
    out = np.zeros((z_max - z_min, y_max - y_min, x_max - x_min), dtype="<u4")

    for iz in range(z_min // cz, -(-z_max // cz)):
        for iy in range(y_min // cy, -(-y_max // cy)):
            for ix in range(x_min // cx, -(-x_max // cx)):
                name = f"{LABEL_ARRAY_PATH}/{iz}.{iy}.{ix}"
                if name not in zf.namelist():
                    continue
                chunk = np.frombuffer(codec.decode(zf.read(name)), dtype="<u4").reshape(LABEL_CHUNK_SHAPE)

                cz0, cy0, cx0 = iz * cz, iy * cy, ix * cx
                sz = slice(max(z_min, cz0) - cz0, min(z_max, cz0 + cz) - cz0)
                sy = slice(max(y_min, cy0) - cy0, min(y_max, cy0 + cy) - cy0)
                sx = slice(max(x_min, cx0) - cx0, min(x_max, cx0 + cx) - cx0)
                oz = slice(max(z_min, cz0) - z_min, min(z_max, cz0 + cz) - z_min)
                oy = slice(max(y_min, cy0) - y_min, min(y_max, cy0 + cy) - y_min)
                ox = slice(max(x_min, cx0) - x_min, min(x_max, cx0 + cx) - x_min)
                out[oz, oy, ox] = chunk[sz, sy, sx]

    return out


def get_mitonet_predicted_kidney_data(
    path: Union[os.PathLike, str], bounding_box: Tuple[int, int, int, int, int, int], download: bool = False,
) -> str:
    """Stream a subvolume of the MitoNet-predicted mouse kidney data and cache it as a zarr v3 store.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        bounding_box: The region to fetch as (z_min, z_max, y_min, y_max, x_min, x_max)
            in voxel coordinates at 16 nm isotropic resolution.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        The filepath to the cached zarr store.
    """
    import zarr
    from zarr.codecs import BloscCodec

    os.makedirs(str(path), exist_ok=True)
    zarr_path = os.path.join(str(path), f"kidney_{_bbox_hash(bounding_box)}.zarr")

    root = zarr.open_group(zarr_path, mode="a")
    if "raw" in root and "labels" in root:
        return zarr_path

    if not download:
        raise RuntimeError(f"No cached data found at '{zarr_path}'. Set download=True to stream it.")

    z_min, z_max, y_min, y_max, x_min, x_max = bounding_box
    for bound, limit in zip((z_max, y_max, x_max), FULL_SHAPE):
        assert bound <= limit, f"Bounding box {bounding_box} exceeds the full volume shape {FULL_SHAPE}"

    raw_block = _read_raw_block(z_min, z_max, y_min, y_max, x_min, x_max)
    label_block = _read_label_block(z_min, z_max, y_min, y_max, x_min, x_max)
    assert raw_block.shape == label_block.shape, f"Shape mismatch: {raw_block.shape} vs {label_block.shape}"

    def _make_array(name, data, shuffle):
        arr = root.create_array(
            name, shape=data.shape, chunks=(64, 256, 256), dtype=data.dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        arr[:] = data

    root.attrs["bounding_box"] = list(bounding_box)
    root.attrs["resolution_nm"] = [16, 16, 16]
    root.attrs["labels_are_automatic_predictions"] = True

    _make_array("raw", raw_block, shuffle="shuffle")
    _make_array("labels", label_block, shuffle="bitshuffle")

    return zarr_path


def get_mitonet_predicted_kidney_paths(
    path: Union[os.PathLike, str],
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    download: bool = False,
) -> List[str]:
    """Get paths to cached MitoNet-predicted mouse kidney zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        bounding_boxes: List of regions to fetch, each as
            (z_min, z_max, y_min, y_max, x_min, x_max) in voxel coordinates at 16 nm resolution.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    return [get_mitonet_predicted_kidney_data(path, bbox, download) for bbox in bounding_boxes]


def get_mitonet_predicted_kidney_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the MitoNet-predicted mouse kidney dataset for mitochondria instance segmentation.

    The labels are automatically generated MitoNet predictions, not manual ground truth.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_boxes: List of subvolumes to use, each as
            (z_min, z_max, y_min, y_max, x_min, x_max) in 16 nm voxel coordinates.
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_mitonet_predicted_kidney_paths(path, bounding_boxes, download)
    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_mitonet_predicted_kidney_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for mitochondria instance segmentation in the MitoNet-predicted mouse kidney data.

    The labels are automatically generated MitoNet predictions, not manual ground truth.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        bounding_boxes: List of subvolumes to use, each as
            (z_min, z_max, y_min, y_max, x_min, x_max) in 16 nm voxel coordinates.
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mitonet_predicted_kidney_dataset(path, patch_shape, bounding_boxes, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
