"""The BBBC030 dataset contains 60 Differential Interference Contrast (DIC) images
of Chinese Hamster Ovary (CHO) cells acquired during initial cell attachment, with
hand-segmented cell contour ground truth annotations.

Raw images are RGB-encoded grayscale (R=G=B). Ground truth files are contour/boundary
maps (thin cell outlines), which are converted to instance segmentation labels by
finding the enclosed regions and labeling them with connected components.

The dataset is located at https://bbbc.broadinstitute.org/BBBC030.
This dataset is from the following publication:
- Koos et al. (2016): https://doi.org/10.1371/journal.pone.0163431
Please cite it if you use this dataset in your research.
"""

import os
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


IMAGE_URL = "https://data.broadinstitute.org/bbbc/BBBC030/images.zip"
IMAGE_CHECKSUM = None

GT_URL = "https://data.broadinstitute.org/bbbc/BBBC030/ground_truth.zip"
GT_CHECKSUM = None


def _contours_to_instances(contour_mask: np.ndarray) -> np.ndarray:
    """Convert a contour/boundary map to an instance segmentation label image.

    Cells are identified as enclosed regions surrounded by boundary pixels.
    The large background region is removed; remaining connected components are
    each assigned a unique integer label.
    """
    from skimage.morphology import binary_dilation, disk
    from skimage.measure import label, regionprops

    boundaries = contour_mask > 0

    # Dilate slightly to close small gaps in hand-drawn contours.
    closed = binary_dilation(boundaries, disk(2))

    # Enclosed interior regions are the complement of the closed boundaries.
    interior = ~closed
    labeled = label(interior)

    # The largest connected component is the background — remove it.
    props = regionprops(labeled)
    if not props:
        return np.zeros_like(contour_mask, dtype=np.int32)

    bg_label = max(props, key=lambda p: p.area).label
    labeled[labeled == bg_label] = 0

    return labeled.astype(np.int32)


def _preprocess(data_dir: str) -> str:
    """Convert raw PNGs to preprocessed H5 files (grayscale raw + instance labels)."""
    import h5py

    h5_dir = os.path.join(data_dir, "h5_data")
    if os.path.exists(h5_dir):
        return h5_dir
    os.makedirs(h5_dir, exist_ok=True)

    raw_paths = natsorted(glob(os.path.join(data_dir, "images", "*.png")))
    for raw_path in tqdm(raw_paths, desc="Preprocessing BBBC030"):
        fname = os.path.splitext(os.path.basename(raw_path))[0]
        h5_path = os.path.join(h5_dir, fname + ".h5")

        gt_path = os.path.join(data_dir, "ground_truth", os.path.basename(raw_path))
        if not os.path.exists(gt_path):
            continue

        raw = imageio.imread(raw_path)
        if raw.ndim == 3:  # grayscale saved as RGB
            raw = raw[..., 0]

        contours = imageio.imread(gt_path)
        instances = _contours_to_instances(contours)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=instances, compression="gzip")

    return h5_dir


def get_bbbc030_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download and preprocess the BBBC030 dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the preprocessed H5 data directory.
    """
    data_dir = os.path.join(path, "BBBC030")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        img_zip = os.path.join(path, "BBBC030_images.zip")
        gt_zip = os.path.join(path, "BBBC030_ground_truth.zip")
        util.download_source(img_zip, IMAGE_URL, download, checksum=IMAGE_CHECKSUM)
        util.download_source(gt_zip, GT_URL, download, checksum=GT_CHECKSUM)
        util.unzip(img_zip, data_dir)
        util.unzip(gt_zip, data_dir)

    return _preprocess(data_dir)


def get_bbbc030_paths(
    path: Union[os.PathLike, str],
    split: Optional[str] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the BBBC030 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. One of 'train', 'val', 'test', or None (use all).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data (H5, key 'raw').
        List of filepaths for the label data (H5, key 'labels').
    """
    h5_dir = get_bbbc030_data(path, download)
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


def get_bbbc030_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Optional[str] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the BBBC030 dataset for DIC cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train', 'val', 'test', or None (use all).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_bbbc030_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="raw",
        label_paths=label_paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_bbbc030_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Optional[str] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the BBBC030 dataloader for DIC cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train', 'val', 'test', or None (use all).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bbbc030_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
