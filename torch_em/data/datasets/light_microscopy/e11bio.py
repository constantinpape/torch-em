"""The E11bio PRISM dataset contains multi-channel expansion microscopy images of mouse
hippocampal CA3 tissue with dense neuron instance segmentation.

The data was generated using PRISM technology: viral barcoding combined with expansion
microscopy and iterative immunolabeling. The tissue is physically expanded ~5× and imaged
across 10 - 18 fluorescent channels (varying per crop) encoding combinatorial protein barcodes
for single-neuron reconstruction.

Voxel resolution (after expansion): ~35 x 35 x 80 nm (xy / z).

Pre-packaged training crops are available on S3 in two flavours:
  - 'instance': 14 crops with dense neuron instance segmentation labels.
  - 'semantic': 17 crops with semantic segmentation labels.

Each channel is stored as a separate (Z, Y, X) dataset under 'raw/ch_00', 'raw/ch_01', ...
Labels are stored as (Z, Y, X) uint32. When raw spatial dimensions exceed labels, the raw
is offset-aligned (center-crop) to match.

Channel counts per crop:
  - crops 0 - 4: 18 channels
  - crops 5 - 11: 12 channels
  - crop 12: 10 channels
  - crop 13: 11 channels
Specify a consistent channel when mixing crops from different groups.

The data is hosted at s3://e11bio-prism (anonymous access, no credentials required).
The dataset is described in the E11bio open-data repository: https://github.com/e11bio/e11-open-data
Please cite this resource if you use the dataset in your research.

NOTE: accessing this dataset requires the `s3fs` package (pip install s3fs).
"""

import os
from typing import List, Literal, Optional, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


S3_BASE = "e11bio-prism/ls/models/training_data"

SPLIT_NUM_CROPS = {
    "instance": 14,
    "semantic": 17,
}


def _get_store(split, crop_id):
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    return s3fs.S3Map(f"{S3_BASE}/{split}/crop_{crop_id}.zarr", s3=fs)


def get_e11bio_data(
    path: Union[os.PathLike, str],
    split: Literal["instance", "semantic"] = "instance",
    crop_ids: Optional[List[int]] = None,
    download: bool = False,
) -> List[str]:
    """Download and cache E11bio PRISM training crops as HDF5 files.

    Each HDF5 file contains:
      - raw/ch_00, raw/ch_01, ...: one (Z, Y, X) uint8 dataset per channel.
      - labels: (Z, Y, X) uint32 instance or semantic segmentation.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        split: Which training split to use. Either 'instance' (14 crops, neuron instance
            segmentation) or 'semantic' (17 crops, semantic segmentation).
        crop_ids: Which crop indices to use. Defaults to all crops for the given split.
        download: Whether to download the data if not already present.

    Returns:
        List of filepaths to the cached HDF5 files.
    """
    import h5py
    import zarr
    from skimage.segmentation import relabel_sequential

    if split not in SPLIT_NUM_CROPS:
        raise ValueError(f"split must be one of {list(SPLIT_NUM_CROPS)}, got {split!r}")

    if crop_ids is None:
        crop_ids = list(range(SPLIT_NUM_CROPS[split]))

    split_dir = os.path.join(path, split)
    os.makedirs(split_dir, exist_ok=True)

    h5_paths = []
    for crop_id in crop_ids:
        h5_path = os.path.join(split_dir, f"crop_{crop_id}.h5")
        h5_paths.append(h5_path)

        if os.path.exists(h5_path):
            continue

        if not download:
            raise RuntimeError(
                f"No cached data found at '{h5_path}'. Set download=True to stream it from S3."
            )

        try:
            import s3fs  # noqa
        except ImportError:
            raise ImportError(
                "The 's3fs' package is required to access the E11bio dataset. "
                "Install it with: 'pip install s3fs'."
            )

        print(f"Streaming E11bio PRISM {split} crop_{crop_id} from S3 ...")
        store = _get_store(split, crop_id)
        f = zarr.open(store, mode="r")

        raw_arr = f["raw"][:]  # (C, Z, Y, X)
        labels_arr = f["labels"][:]  # (Z, Y, X)

        # Align raw spatially to labels using the stored offsets.
        raw_offset = f["raw"].attrs.get("offset", [0, 0, 0])
        lbl_offset = f["labels"].attrs.get("offset", [0, 0, 0])
        resolution = f["raw"].attrs.get("resolution", [1, 1, 1])

        z0 = round((lbl_offset[0] - raw_offset[0]) / resolution[0])
        y0 = round((lbl_offset[1] - raw_offset[1]) / resolution[1])
        x0 = round((lbl_offset[2] - raw_offset[2]) / resolution[2])

        lz, ly, lx = labels_arr.shape
        raw_arr = raw_arr[:, z0:z0 + lz, y0:y0 + ly, x0:x0 + lx]

        # Relabel to consecutive integers.
        labels_arr, _, _ = relabel_sequential(labels_arr)

        with h5py.File(h5_path, "w", locking=False) as out:
            out.attrs["crop_id"] = crop_id
            out.attrs["split"] = split
            out.attrs["num_channels"] = raw_arr.shape[0]
            raw_grp = out.create_group("raw")
            for ch_idx, ch_data in enumerate(raw_arr):
                raw_grp.create_dataset(
                    f"ch_{ch_idx:02d}", data=ch_data.astype("uint8"), compression="gzip", chunks=True
                )
            out.create_dataset("labels", data=labels_arr.astype("uint32"), compression="gzip", chunks=True)

        print(f"Cached to {h5_path}  ({raw_arr.shape[0]} channels, spatial {labels_arr.shape})")

    return h5_paths


def get_e11bio_paths(
    path: Union[os.PathLike, str],
    split: Literal["instance", "semantic"] = "instance",
    crop_ids: Optional[List[int]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the E11bio PRISM HDF5 cache files.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        split: Which training split to use. Either 'instance' or 'semantic'.
        crop_ids: Which crop indices to use. Defaults to all crops for the given split.
        download: Whether to download the data if not already present.

    Returns:
        List of filepaths to the cached HDF5 files.
    """
    return get_e11bio_data(path, split, crop_ids, download)


def get_e11bio_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    split: Literal["instance", "semantic"] = "instance",
    crop_ids: Optional[List[int]] = None,
    channel: int = 0,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> Dataset:
    """Get the E11bio PRISM dataset for neuron instance or semantic segmentation.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        split: Which training split to use. Either 'instance' (14 crops) or
            'semantic' (17 crops).
        crop_ids: Which crop indices to use. Defaults to all crops for the given split.
        channel: Which fluorescence channel to use as raw input (default 0).
            Channel counts vary per crop (10 - 18); use a channel index present in all
            selected crops (0 - 9 is safe for all crops).
        download: Whether to download the data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_e11bio_paths(path, split, crop_ids, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=False, boundaries=boundaries, offsets=offsets
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key=f"raw/ch_{channel:02d}",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        ndim=3,
        **kwargs,
    )


def get_e11bio_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    split: Literal["instance", "semantic"] = "instance",
    crop_ids: Optional[List[int]] = None,
    channel: int = 0,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for neuron instance or semantic segmentation in the E11bio PRISM dataset.

    Args:
        path: Filepath to a folder where the cached HDF5 files will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        split: Which training split to use. Either 'instance' (14 crops) or
            'semantic' (17 crops).
        crop_ids: Which crop indices to use. Defaults to all crops for the given split.
        channel: Which fluorescence channel to use as raw input (default 0).
            Channel counts vary per crop (10 - 18); use a channel index present in all
            selected crops (0 - 9 is safe for all crops).
        download: Whether to download the data if not already present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_e11bio_dataset(
        path, patch_shape, split, crop_ids, channel, download, offsets, boundaries, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
