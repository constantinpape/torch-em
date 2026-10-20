"""ASEM is a dataset for segmentation of cellular structures in FIB-SEM.

The data is hosted in the public `asem-project` S3 bucket (https://open.quiltdata.com/b/asem-project).
Each cell is stored as `datasets/<id>/<id>.zarr`, with the raw data under `volumes/raw` and the ground-truth
annotations under `volumes/labels/<organelle>`. The labels are semantic (a single foreground value).
Some label arrays span the whole cell, others are crops with an offset into the raw volume;
only the raw region covered by the labels of the chosen organelles is downloaded.
A bounding box (z0, z1, y0, y1, x0, x1) in raw voxels can be given per volume. Then only this region is
downloaded, into a separate store `<id>/<id>_z<z0>-<z1>_y<y0>-<y1>_x<x0>-<x1>.zarr` with the box in its attributes.

The dataset was publised in https://doi.org/10.1083/jcb.202208005.
Please cite this publication if you use the dataset in your research.
"""

import os
from concurrent import futures
from typing import Dict, Union, Tuple, Optional, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util
from ... import ConcatDataset


BUCKET = "s3://asem-project/datasets"

# The volume ids are the ids in the bucket. They correspond to the cells of the publication as follows:
# 46 and 58 are cells 1 and 2 (HEK293A, the bucket does not say which is which), 61 is cell 3 (BSC-1),
# 64 is cell 6 (SUM159), and 1E, 2E and 3A are cells 12, 13 and 13a (high-pressure frozen).
VOLUMES = ["46", "58", "61", "64", "1E", "2E", "3A"]

ORGANELLES = {
    "mito": ["46", "58", "61", "64", "2E", "3A"],
    "golgi": ["46", "58", "61", "64"],
    "er": ["46", "58", "61", "64"],
    "ccp": ["1E", "2E"],
    "np": ["3A"],
    "np_bottom": ["3A"],
}

# The mito labels of 61 span the whole cell, but only this block (in voxels of the label array) is annotated.
# The block is the hold-out region the publication evaluates on (the extent of `volumes/predictions/mito`).
ANNOTATED_REGIONS = {("61", "mito"): np.s_[0:700, 190:730, 2630:3380]}

BLOCK_SHAPE = (128, 512, 512)
N_THREADS = 8


def _check_volume_ids(volume_ids, organelle):
    for volume_id in volume_ids:
        if volume_id.startswith("cell_"):
            raise ValueError(
                f"The ASEM volume ids are the ids of the S3 bucket, for example '64' instead of 'cell_6', "
                f"got '{volume_id}'. See the comment above 'VOLUMES' for the cell numbers of the publication."
            )
        if volume_id not in ORGANELLES[organelle]:
            raise ValueError(
                f"The volume '{volume_id}' has no '{organelle}' labels. Choose from {ORGANELLES[organelle]}."
            )


def _open_remote(volume_id, key):
    import fsspec
    import zarr

    store = fsspec.get_mapper(f"{BUCKET}/{volume_id}/{volume_id}.zarr/volumes/{key}", anon=True)
    return zarr.open_array(store, mode="r", zarr_format=2)


def _copy_region(src, dst, region):
    """Copy a region of the remote array into the local array, block by block."""
    starts = [r.start for r in region]
    grid = [int(np.ceil(s / b)) for s, b in zip(dst.shape, BLOCK_SHAPE)]
    blocks = [
        tuple(slice(i * b, min((i + 1) * b, s)) for i, b, s in zip(index, BLOCK_SHAPE, dst.shape))
        for index in np.ndindex(*grid)
    ]

    def copy_block(block):
        dst[block] = src[tuple(slice(start + b.start, start + b.stop) for start, b in zip(starts, block))]

    with futures.ThreadPoolExecutor(N_THREADS) as tp:
        list(tp.map(copy_block, blocks))


def _get_raw_key(volume_path, organelle):
    return f"raw_{organelle}" if os.path.exists(os.path.join(volume_path, "volumes", f"raw_{organelle}")) else "raw"


def _have_organelle(volume_path, organelle):
    volumes = os.path.join(volume_path, "volumes")
    labels = os.path.join(volumes, "labels", organelle)
    return os.path.exists(labels) and os.path.exists(os.path.join(volumes, _get_raw_key(volume_path, organelle)))


def _get_volume_path(path, volume_id, bounding_box):
    if bounding_box is None:
        return os.path.join(path, volume_id, f"{volume_id}.zarr")
    z0, z1, y0, y1, x0, x1 = bounding_box
    return os.path.join(path, volume_id, f"{volume_id}_z{z0}-{z1}_y{y0}-{y1}_x{x0}-{x1}.zarr")


def _download_organelle(volume_path, volume_id, organelle, bounding_box=None):
    """Download the labels of one organelle and the raw region they cover.

    The raw data is stored as 'volumes/raw' if the labels cover the whole raw volume (shared by all organelles),
    and as 'volumes/raw_<organelle>' otherwise. If a bounding box is given, the raw data and the labels of this
    box are stored, and the raw data is always 'volumes/raw'. Each array is written to a temporary folder and
    moved in place once it is complete, so that an interrupted download is not mistaken for a complete one.
    """
    import zarr
    from zarr.codecs import BloscCodec

    raw, labels = _open_remote(volume_id, "raw"), _open_remote(volume_id, f"labels/{organelle}")
    resolution = np.array(raw.attrs["resolution"])
    offset = (np.array(labels.attrs["offset"]) - np.array(raw.attrs["offset"])) // resolution
    if bounding_box is None:
        label_region = ANNOTATED_REGIONS.get((volume_id, organelle), tuple(slice(0, s) for s in labels.shape))
        raw_region = tuple(slice(int(o) + r.start, int(o) + r.stop) for o, r in zip(offset, label_region))
    else:
        raw_region = tuple(slice(start, stop) for start, stop in zip(bounding_box[::2], bounding_box[1::2]))
        label_region = tuple(slice(r.start - int(o), r.stop - int(o)) for o, r in zip(offset, raw_region))
        if any(r.start < 0 or r.stop > s or r.start >= r.stop for r, s in zip(label_region, labels.shape)):
            label_box = [(int(o), int(o) + s) for o, s in zip(offset, labels.shape)]
            raise ValueError(
                f"The bounding box {bounding_box} is not inside the '{organelle}' labels of the ASEM volume "
                f"'{volume_id}', which cover the raw region {label_box}."
            )
    full_raw = all(r.start == 0 and r.stop == s for r, s in zip(raw_region, raw.shape))
    raw_key = "raw" if full_raw or bounding_box is not None else f"raw_{organelle}"

    root = zarr.open_group(volume_path, mode="a")
    if bounding_box is not None:
        root.attrs["bounding_box"] = list(bounding_box)
    group = root.require_group("volumes")
    for key, src, region in [(raw_key, raw, raw_region), (f"labels/{organelle}", labels, label_region)]:
        out_path = os.path.join(volume_path, "volumes", key)
        if os.path.exists(out_path):
            continue
        print(f"Downloading 'volumes/{key}' of the ASEM volume '{volume_id}' ...")
        tmp_key = f"{key}_incomplete"
        dst = group.create_array(
            tmp_key, shape=tuple(r.stop - r.start for r in region), chunks=(64, 256, 256), dtype=src.dtype,
            compressors=BloscCodec(cname="zstd", clevel=5), overwrite=True,
        )
        _copy_region(src, dst, region)
        dst.attrs.update({
            "resolution": resolution.tolist(),
            "offset": (np.array(raw.attrs["offset"]) + np.array([r.start for r in raw_region]) * resolution).tolist(),
        })
        os.replace(os.path.join(volume_path, "volumes", tmp_key), out_path)


def get_asem_data(
    path: Union[os.PathLike, str],
    volume_ids: List[str],
    download: bool = False,
    organelles: Optional[Union[List[str], str]] = None,
    bounding_boxes: Optional[Dict[str, Tuple[int, int, int, int, int, int]]] = None,
):
    """Download the ASEM dataset.

    The data is streamed from the public S3 bucket at s3://asem-project, which requires `s3fs`.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        volume_ids: List of volumes to download.
        download: Whether to download the data if it is not present.
        organelles: The organelles to download for each volume. By default all organelles annotated in the volume.
        bounding_boxes: The bounding box per volume id, as (z0, z1, y0, y1, x0, x1) in raw voxels.
            Only this region of the volume is downloaded. Volumes without a bounding box are downloaded in full.
    """
    if organelles is None:
        organelles = list(ORGANELLES)
    elif isinstance(organelles, str):
        organelles = [organelles]

    for volume_id in volume_ids:
        volume_organelles = [organelle for organelle in organelles if volume_id in ORGANELLES[organelle]]
        for organelle in volume_organelles:
            _check_volume_ids([volume_id], organelle)
        if not volume_organelles:
            raise ValueError(f"The volume '{volume_id}' has none of the organelles {organelles}.")

        bounding_box = None if bounding_boxes is None else bounding_boxes.get(volume_id)
        volume_path = _get_volume_path(path, volume_id, bounding_box)
        for organelle in volume_organelles:
            if _have_organelle(volume_path, organelle):
                continue
            if not download:
                raise FileNotFoundError(
                    f"The '{organelle}' data of the ASEM volume '{volume_id}' is not found at '{volume_path}', "
                    "and 'download' is set to False."
                )
            _download_organelle(volume_path, volume_id, organelle, bounding_box)


def get_asem_paths(
    path: Union[os.PathLike, str],
    volume_ids: List[str],
    download: bool = False,
    organelles: Optional[Union[List[str], str]] = None,
    bounding_boxes: Optional[Dict[str, Tuple[int, int, int, int, int, int]]] = None,
) -> List[str]:
    """Get paths to the ASEM data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        volume_ids: List of volumes to download.
        download: Whether to download the data if it is not present.
        organelles: The organelles to download for each volume. By default all organelles annotated in the volume.
        bounding_boxes: The bounding box per volume id, as (z0, z1, y0, y1, x0, x1) in raw voxels.
            Only this region of the volume is downloaded. Volumes without a bounding box are downloaded in full.

    Returns:
        List of paths for all volume ids.
    """
    get_asem_data(path, volume_ids, download, organelles, bounding_boxes)
    bounding_boxes = {} if bounding_boxes is None else bounding_boxes
    return [_get_volume_path(path, volume_id, bounding_boxes.get(volume_id)) for volume_id in volume_ids]


def _check_input_args(input_arg, default_values):
    if input_arg is None:
        input_arg = default_values
    else:
        if isinstance(input_arg, str):
            assert input_arg in default_values
            input_arg = [input_arg]

    return input_arg


def get_asem_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    download: bool = False,
    organelles: Optional[Union[List[str], str]] = None,
    volume_ids: Optional[Union[List[str], str]] = None,
    bounding_boxes: Optional[Dict[str, Tuple[int, int, int, int, int, int]]] = None,
    **kwargs
) -> Dataset:
    """Get dataset for segmentation of organelles in FIB-SEM cells.

    The labels are semantic, use a label transform (e.g. connected components) to obtain instances.
    For the mito labels of volume 61 only the annotated block is used (see `ANNOTATED_REGIONS`),
    so any `rois` passed for it are relative to that block. A bounding box replaces this block.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        organelles: The choice of organelles.
        volume_ids: The choice of volumes.
        bounding_boxes: The bounding box per volume id, as (z0, z1, y0, y1, x0, x1) in raw voxels.
            Only this region of the volume is used. Volumes without a bounding box are used in full.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    organelles = _check_input_args(organelles, ORGANELLES)

    all_datasets = []
    for organelle in organelles:
        if volume_ids is None:
            organelle_volume_ids = ORGANELLES[organelle]
        else:
            organelle_volume_ids = [volume_ids] if isinstance(volume_ids, str) else list(volume_ids)
            _check_volume_ids(organelle_volume_ids, organelle)

        volume_paths = get_asem_paths(path, organelle_volume_ids, download, [organelle], bounding_boxes)
        for volume_path in volume_paths:
            dataset = torch_em.default_segmentation_dataset(
                raw_paths=volume_path,
                raw_key=f"volumes/{_get_raw_key(volume_path, organelle)}",
                label_paths=volume_path,
                label_key=f"volumes/labels/{organelle}",
                patch_shape=patch_shape,
                is_seg_dataset=True,
                **kwargs
            )
            dataset.max_sampling_attempts = 5000
            all_datasets.append(dataset)

    return ConcatDataset(*all_datasets)


def get_asem_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    download: bool = False,
    organelles: Optional[Union[List[str], str]] = None,
    volume_ids: Optional[Union[List[str], str]] = None,
    bounding_boxes: Optional[Dict[str, Tuple[int, int, int, int, int, int]]] = None,
    **kwargs
) -> DataLoader:
    """Get dataloader for the segmentation of organelles in FIB-SEM cells.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        organelles: The choice of organelles.
        volume_ids: The choice of volumes.
        bounding_boxes: The bounding box per volume id, as (z0, z1, y0, y1, x0, x1) in raw voxels.
            Only this region of the volume is used. Volumes without a bounding box are used in full.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_asem_dataset(path, patch_shape, download, organelles, volume_ids, bounding_boxes, **ds_kwargs)
    loader = torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
    return loader
