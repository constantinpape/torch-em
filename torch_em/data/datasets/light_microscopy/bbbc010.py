"""The BBBC010 dataset contains brightfield and GFP images of the roundworm
C. elegans from a live/dead assay screen for novel anti-infectives. Animals were
exposed to the pathogen Enterococcus faecalis and either treated with ampicillin
("live" phenotype) or left untreated ("dead" phenotype).

The dataset contains 100 wells from a 384-well plate, each imaged in two channels
(w1: brightfield, w2: GFP), with instance segmentation ground truth for individual
worms.

The dataset is located at https://bbbc.broadinstitute.org/BBBC010.
This dataset is CC0 (public domain). If you use it, please cite it as:
"We used the C. elegans infection live/dead image set version 1 provided by Fred
Ausubel and available from the Broad Bioimage Benchmark Collection [Ljosa et al.,
Nature Methods, 2012]." https://doi.org/10.1038/nmeth.2083
"""

import os
import re
from glob import glob
from natsort import natsorted
from typing import List, Optional, Tuple, Union

import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


IMAGE_URL = "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v2_images.zip"
IMAGE_CHECKSUM = None

GT_URL = "https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip"
GT_CHECKSUM = None

WELL_PATTERN = re.compile(r"_([A-E]\d{2})_w(\d)_")


def _get_well_and_channel(fname: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract the well id (e.g. 'A01') and channel number (1 or 2) from a raw image filename."""
    match = WELL_PATTERN.search(fname)
    if match is None:
        return None, None
    return match.group(1), int(match.group(2))


def _merge_worm_masks(mask_paths: List[str]) -> np.ndarray:
    """Merge per-worm binary masks into a single instance segmentation label image."""
    mask_paths = natsorted(mask_paths)
    ref = imageio.imread(mask_paths[0])
    instances = np.zeros(ref.shape, dtype=np.int32)
    for i, mask_path in enumerate(mask_paths, start=1):
        mask = imageio.imread(mask_path) > 0
        instances[mask] = i
    return instances


def _preprocess(data_dir: str, channel: int) -> str:
    """Convert raw TIFs and per-worm ground truth PNGs to preprocessed H5 files."""
    import h5py

    h5_dir = os.path.join(data_dir, f"h5_data_w{channel}")
    if os.path.exists(h5_dir):
        return h5_dir
    os.makedirs(h5_dir, exist_ok=True)

    raw_paths = glob(os.path.join(data_dir, "images", "*.tif"))
    well_to_raw = {}
    for raw_path in raw_paths:
        well, this_channel = _get_well_and_channel(os.path.basename(raw_path))
        if well is None or this_channel != channel:
            continue
        well_to_raw[well] = raw_path

    gt_dir = os.path.join(data_dir, "BBBC010_v1_foreground_eachworm")
    for well, raw_path in tqdm(sorted(well_to_raw.items()), desc="Preprocessing BBBC010"):
        mask_paths = glob(os.path.join(gt_dir, f"{well}_*_ground_truth.png"))
        if len(mask_paths) == 0:
            continue

        raw = imageio.imread(raw_path)
        instances = _merge_worm_masks(mask_paths)

        h5_path = os.path.join(h5_dir, f"{well}.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=instances, compression="gzip")

    return h5_dir


def get_bbbc010_data(path: Union[os.PathLike, str], channel: int = 1, download: bool = False) -> str:
    """Download and preprocess the BBBC010 dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        channel: The imaging channel to use as raw input. Default: 1 (brightfield).
            Available channels: 1=brightfield, 2=GFP.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the preprocessed H5 data directory.
    """
    data_dir = os.path.join(path, "BBBC010")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        img_zip = os.path.join(path, "BBBC010_v2_images.zip")
        gt_zip = os.path.join(path, "BBBC010_v1_foreground_eachworm.zip")
        util.download_source(img_zip, IMAGE_URL, download, checksum=IMAGE_CHECKSUM)
        util.download_source(gt_zip, GT_URL, download, checksum=GT_CHECKSUM)
        util.unzip(img_zip, os.path.join(data_dir, "images"))
        util.unzip(gt_zip, data_dir)

    return _preprocess(data_dir, channel)


def get_bbbc010_paths(
    path: Union[os.PathLike, str],
    split: Optional[str] = None,
    channel: int = 1,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the BBBC010 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. One of 'train', 'val', 'test', or None (use all).
        channel: The imaging channel to use as raw input. Default: 1 (brightfield).
            Available channels: 1=brightfield, 2=GFP.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data (H5, key 'raw').
        List of filepaths for the label data (H5, key 'labels').
    """
    h5_dir = get_bbbc010_data(path, channel, download)
    h5_paths = natsorted(glob(os.path.join(h5_dir, "*.h5")))

    if len(h5_paths) == 0:
        raise RuntimeError(f"No preprocessed files found in {h5_dir}.")

    if split is None:
        return h5_paths, h5_paths

    train_paths, test_paths = train_test_split(h5_paths, test_size=0.2, random_state=42)
    train_paths, val_paths = train_test_split(train_paths, test_size=0.15, random_state=42)

    split_map = {"train": train_paths, "val": val_paths, "test": test_paths}
    assert split in split_map, f"'{split}' is not a valid split. Choose from {list(split_map)}."
    selected = split_map[split]
    return selected, selected


def get_bbbc010_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Optional[str] = None,
    channel: int = 1,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the BBBC010 dataset for C. elegans instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train', 'val', 'test', or None (use all).
        channel: The imaging channel to use as raw input. Default: 1 (brightfield).
            Available channels: 1=brightfield, 2=GFP.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_bbbc010_paths(path, split, channel, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="raw",
        label_paths=label_paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_bbbc010_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Optional[str] = None,
    channel: int = 1,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the BBBC010 dataloader for C. elegans instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train', 'val', 'test', or None (use all).
        channel: The imaging channel to use as raw input. Default: 1 (brightfield).
            Available channels: 1=brightfield, 2=GFP.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bbbc010_dataset(path, patch_shape, split, channel, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
