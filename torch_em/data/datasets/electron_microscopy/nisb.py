"""NISB is a large-scale synthetic benchmark for neuron instance segmentation in connectomics.

It comprises 9 settings with varying difficulty and imaging conditions, each providing 5 training
cubes, 1 validation cube, and 1 test cube. The train_100 setting is an exception with 100 training
cubes for scaling analysis. Cubes are 27µm side length at 9x9x20 nm voxel size (liconn: 9x9x12 nm).
The multichannel setting stores 8-channel embeddings instead of a single grayscale image.

Data is streamed directly from S3 via s3fs and written to local zarr v3 stores (chunk 64^3,
shard 512^3, zstd compression) with (z, y, x) axis order. The source is zarr v2 with (x, y, z)
axis order; spatial axes are transposed and the trailing singleton channel dim on img is squeezed
during the write. Requires s3fs (pip install s3fs).

The data is described in https://doi.org/10.17617/1.r2mm-1h33.
Please cite it if you use this dataset for a publication.
"""

import os
import shutil
import warnings
from glob import glob
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import torch_em
from .. import util


NISB_S3_ENDPOINT = "https://s3.nexus.mpcdf.mpg.de:443"
NISB_S3_BUCKET = "nisb"

NISB_SETTINGS = [
    "base", "train_100", "slice_perturbed", "pos_guidance", "neg_guidance",
    "no_touch_thick", "touching_thin", "liconn", "multichannel",
]

NISB_CHUNK_SHAPE = (64, 64, 64)
NISB_SHARD_SHAPE = (512, 512, 512)


def _nisb_n_seeds(setting: str, split: str) -> int:
    if split in ("val", "test"):
        return 1
    return 100 if setting == "train_100" else 5


def _nisb_zarr_complete(zarr_path: str) -> bool:
    return (
        os.path.isfile(os.path.join(zarr_path, "zarr.json"))
        and os.path.isdir(os.path.join(zarr_path, "img"))
        and os.path.isdir(os.path.join(zarr_path, "seg"))
    )


def _nisb_create_v3_array(root, name, shape, dtype, is_label):
    from zarr.codecs import BloscCodec
    shuffle = "bitshuffle" if (np.issubdtype(np.dtype(dtype), np.integer) and is_label) else "shuffle"
    chunks = NISB_CHUNK_SHAPE + tuple(shape[3:])
    shards = NISB_SHARD_SHAPE + tuple(shape[3:])
    return root.create_array(
        name, shape=shape, chunks=chunks, shards=shards, dtype=dtype,
        compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
    )


def _nisb_write_cube_v3(src, v3_path: str) -> None:
    """Stream a NISB cube from a zarr v2 source to a local zarr v3 store.

    Transposes axes from (x, y, z) to (z, y, x) and squeezes the trailing singleton
    channel dimension on the image array.
    """
    import zarr

    img_v2 = src["img"]
    seg_v2 = src["seg"]

    squeeze_img = img_v2.ndim == 4 and img_v2.shape[-1] == 1
    if squeeze_img:
        img_shape_v3 = (img_v2.shape[2], img_v2.shape[1], img_v2.shape[0])
    else:
        img_shape_v3 = (img_v2.shape[2], img_v2.shape[1], img_v2.shape[0], img_v2.shape[3])
    seg_shape_v3 = (seg_v2.shape[2], seg_v2.shape[1], seg_v2.shape[0])

    tmp_path = v3_path + ".tmp"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    root = zarr.open_group(tmp_path, mode="w", zarr_format=3)
    img_v3 = _nisb_create_v3_array(root, "img", img_shape_v3, np.dtype("uint8"), False)
    seg_v3 = _nisb_create_v3_array(root, "seg", seg_shape_v3, np.dtype("uint16"), True)

    Z, Y, X = seg_shape_v3
    sz, sy, sx = NISB_SHARD_SHAPE
    for z0 in range(0, Z, sz):
        for y0 in range(0, Y, sy):
            for x0 in range(0, X, sx):
                z1, y1, x1 = min(z0 + sz, Z), min(y0 + sy, Y), min(x0 + sx, X)
                block_img = np.asarray(img_v2[x0:x1, y0:y1, z0:z1])
                if squeeze_img:
                    block_img = block_img[..., 0]
                img_v3[z0:z1, y0:y1, x0:x1] = np.moveaxis(block_img, [0, 2], [2, 0])
                block_seg = np.asarray(seg_v2[x0:x1, y0:y1, z0:z1])
                seg_v3[z0:z1, y0:y1, x0:x1] = block_seg.transpose(2, 1, 0)

    shutil.move(tmp_path, v3_path)


def _nisb_open_remote(setting: str, split: str, seed_idx: int):
    """Open a NISB seed cube from S3 as a zarr v2 group via s3fs."""
    try:
        import s3fs
    except ImportError:
        raise ImportError("The 's3fs' package is required to download NISB data. Install it with: pip install s3fs")
    import zarr

    fs = s3fs.S3FileSystem(anon=True, endpoint_url=NISB_S3_ENDPOINT)
    s3_path = f"{NISB_S3_BUCKET}/{setting}/{split}/seed{seed_idx}/data.zarr"
    store = zarr.storage.FsspecStore(fs=fs, path=s3_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*asynchronous.*")
        return zarr.open_group(store, mode="r", zarr_format=2)


def get_nisb_data(path: Union[os.PathLike, str], setting: str, split: str, download: bool) -> str:
    """Stream and cache NISB data for a given setting and split from S3.

    Data is read from S3 via s3fs and written to local zarr v3 stores with (z, y, x) axis
    order, sharding (chunk 64^3, shard 512^3), and zstd compression. Already-cached seeds
    are skipped on subsequent calls.

    Args:
        path: Filepath to a folder where the cached data will be saved.
        setting: The NISB setting. One of NISB_SETTINGS.
        split: The data split, one of 'train', 'val', 'test'.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        The filepath to the split directory containing seed subdirectories.
    """
    assert setting in NISB_SETTINGS, f"Invalid setting '{setting}'. Choose from {NISB_SETTINGS}."
    assert split in ("train", "val", "test"), f"Invalid split '{split}'. Choose 'train', 'val', or 'test'."

    split_dir = os.path.join(str(path), setting, split)
    n = _nisb_n_seeds(setting, split)

    for i in tqdm(range(n), desc=f"NISB {setting}/{split}", leave=False):
        seed_dir = os.path.join(split_dir, f"seed{i}")
        zarr_path = os.path.join(seed_dir, "data.zarr")

        if _nisb_zarr_complete(zarr_path):
            continue

        if not download:
            raise RuntimeError(
                f"No NISB data for setting '{setting}' split '{split}' seed {i} at '{zarr_path}'. "
                "Set download=True to stream it from S3."
            )

        os.makedirs(seed_dir, exist_ok=True)
        print(f"Streaming NISB {setting}/{split}/seed{i} from S3 ...")
        src = _nisb_open_remote(setting, split, i)
        _nisb_write_cube_v3(src, zarr_path)

    return split_dir


def get_nisb_paths(
    path: Union[os.PathLike, str],
    setting: str = "base",
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
) -> List[str]:
    """Get paths to NISB zarr stores for a given setting and split.

    Args:
        path: Filepath to a folder where the cached data is saved.
        setting: The NISB setting. One of NISB_SETTINGS.
        split: The data split, one of 'train', 'val', 'test'.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        Sorted list of filepaths to the zarr stores, one per cube/seed.
    """
    split_dir = get_nisb_data(path, setting, split, download)
    paths = sorted(glob(os.path.join(split_dir, "seed*", "data.zarr")))
    if not paths:
        raise RuntimeError(
            f"No zarr files found in '{split_dir}'. The download may have failed or the directory is empty."
        )
    return paths


def get_nisb_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    setting: str = "base",
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the NISB dataset for neuron instance segmentation in EM.

    NISB provides 9 settings of varying difficulty, each with multiple cubes at 27µm side length.
    Image data is stored under the zarr key 'img' with shape (z, y, x) and segmentation under 'seg'.
    The multichannel setting stores 8-channel data with shape (z, y, x, 8).

    Args:
        path: Filepath to a folder where the cached data will be saved.
        patch_shape: The patch shape to use for training.
        setting: The NISB setting. One of NISB_SETTINGS. Default 'base'.
        split: The data split, one of 'train', 'val', 'test'.
        download: Whether to stream and cache the data if it is not present.
            Requires s3fs (pip install s3fs).
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_nisb_paths(path, setting, split, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="img",
        label_paths=paths,
        label_key="seg",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_nisb_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    setting: str = "base",
    split: Literal["train", "val", "test"] = "train",
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for neuron instance segmentation in the NISB dataset.

    Args:
        path: Filepath to a folder where the cached data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        setting: The NISB setting. One of NISB_SETTINGS. Default 'base'.
        split: The data split, one of 'train', 'val', 'test'.
        download: Whether to stream and cache the data if it is not present.
            Requires s3fs (pip install s3fs).
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_nisb_dataset(
        path=path,
        patch_shape=patch_shape,
        setting=setting,
        split=split,
        download=download,
        offsets=offsets,
        boundaries=boundaries,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
