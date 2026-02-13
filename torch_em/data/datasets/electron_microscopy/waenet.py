"""The WAE-NET dataset contains seven biological electron microscopy datasets
for multi-class semantic segmentation of cellular structures.

The seven sub-datasets cover different cell types and imaging modalities:
    - Dataset 1: Human pancreatic carcinoid cell line (FIB-SEM) — background, cytoplasm, nucleus
    - Dataset 2: BON cell during interphase (ssTEM) — background, cytoplasm, chromosomes
    - Dataset 3: BON cell during mitosis (ssTEM) — background, cytoplasm, nucleus, mitochondria
    - Dataset 4: Human T-cell line Jurkat (TEM) — background, cytoplasm, nucleus
    - Dataset 5: Primary human T-cell blood (TEM) — background, cytoplasm, nucleus
    - Dataset 6: Murine B-cell line J558L (TEM) — background, cytoplasm, nucleus
    - Dataset 7: Phytohemagglutinin/IL-2 expanded human T cells (TEM) — background, cytoplasm, nucleus

The data is available at https://doi.org/10.17632/9rdmnn2x4x.1.
The dataset was published in https://doi.org/10.1007/s00418-022-02148-3.
Please cite this publication if you use the dataset in your research.
"""

import os
from glob import glob
from shutil import rmtree
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URL = "https://zenodo.org/records/6603083/files/Datasets.zip"
CHECKSUM = None

# Maps dataset_id -> number of segmentation classes (including background)
DATASET_CLASSES = {1: 3, 2: 3, 3: 4, 4: 3, 5: 3, 6: 3, 7: 3}

# Maps dataset_id -> ordered list of class names (index 0 = background, etc.)
DATASET_CLASS_NAMES = {
    1: ["background", "cytoplasm", "nucleus"],
    2: ["background", "cytoplasm", "chromosomes"],
    3: ["background", "cytoplasm", "nucleus", "mitochondria"],
    4: ["background", "cytoplasm", "nucleus"],
    5: ["background", "cytoplasm", "nucleus"],
    6: ["background", "cytoplasm", "nucleus"],
    7: ["background", "cytoplasm", "nucleus"],
}


def _get_dataset_dir(data_root, dataset_id):
    """Find the subdirectory for a given dataset ID inside the extracted archive."""
    for dname in (
        f"Dataset {dataset_id}", f"Dataset_{dataset_id}", f"Dataset{dataset_id}", f"D{dataset_id}", str(dataset_id)
    ):
        d = os.path.join(data_root, dname)
        if os.path.exists(d):
            return d
    raise RuntimeError(
        f"Cannot find a sub-directory for dataset {dataset_id} inside '{data_root}'. "
        f"Contents: {os.listdir(data_root)}"
    )


def _get_image_mask_dirs(dataset_dir):
    """Find the image and mask subdirectories within a per-dataset directory."""
    img_dir = None
    for name in ("Images", "images", "Image", "image", "Raw", "raw"):
        candidate = os.path.join(dataset_dir, name)
        if os.path.exists(candidate):
            img_dir = candidate
            break

    mask_dir = None
    for name in ("Ground truth mask", "Masks", "masks", "Mask", "mask", "Labels", "labels", "Label", "label"):
        candidate = os.path.join(dataset_dir, name)
        if os.path.exists(candidate):
            mask_dir = candidate
            break

    if img_dir is None or mask_dir is None:
        raise RuntimeError(
            f"Cannot find image/mask directories inside '{dataset_dir}'. "
            f"Contents: {os.listdir(dataset_dir)}"
        )
    return img_dir, mask_dir


def _create_h5_files(data_root, dataset_id, out_dir):
    """Convert TIF image/mask pairs for one sub-dataset into individual HDF5 files."""
    import h5py
    import imageio.v3 as imageio

    dataset_dir = _get_dataset_dir(data_root, dataset_id)
    img_dir, mask_dir = _get_image_mask_dirs(dataset_dir)

    image_files = sorted(
        glob(os.path.join(img_dir, "*.tif")) +
        glob(os.path.join(img_dir, "*.tiff")) +
        glob(os.path.join(img_dir, "*.png"))
    )
    mask_files = sorted(
        glob(os.path.join(mask_dir, "*.tif")) +
        glob(os.path.join(mask_dir, "*.tiff")) +
        glob(os.path.join(mask_dir, "*.png"))
    )

    assert len(image_files) > 0, f"No TIF files found in '{img_dir}'"
    assert len(image_files) == len(mask_files), (
        f"Mismatch: {len(image_files)} images vs {len(mask_files)} masks in '{dataset_dir}'"
    )

    os.makedirs(out_dir, exist_ok=True)

    for img_path, mask_path in zip(image_files, mask_files):
        fname = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, f"{fname}.h5")

        raw = imageio.imread(img_path)
        if raw.ndim == 3:  # drop extra channels (e.g. RGBA -> grayscale)
            raw = raw[..., 0]

        labels = imageio.imread(mask_path)
        if labels.ndim == 3:
            labels = labels[..., 0]

        # Remap arbitrary grayscale values to consecutive class indices (0, 1, 2, …).
        unique_vals = np.sort(np.unique(labels))
        if not np.array_equal(unique_vals, np.arange(len(unique_vals))):
            new_labels = np.zeros_like(labels)
            for cls_idx, val in enumerate(unique_vals):
                new_labels[labels == val] = cls_idx
            labels = new_labels

        class_names = DATASET_CLASS_NAMES[dataset_id]

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            label_group = f.create_group("labels")
            for cls_idx, cls_name in enumerate(class_names):
                binary_mask = (labels == cls_idx).astype("uint8")
                label_group.create_dataset(cls_name, data=binary_mask, compression="gzip")


def get_waenet_data(path: Union[os.PathLike, str], dataset_id: int, download: bool = False) -> str:
    """Download and preprocess the WAE-NET dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset_id: Which of the seven sub-datasets to use (1–7).
        download: Whether to download the data if it is not present.

    Returns:
        The path to the directory containing the preprocessed HDF5 files.
    """
    if dataset_id not in DATASET_CLASSES:
        raise ValueError(f"Invalid dataset_id {dataset_id!r}. Choose from {sorted(DATASET_CLASSES)}.")

    out_dir = os.path.join(path, f"dataset_{dataset_id}")
    if os.path.exists(out_dir):
        return out_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "Datasets.zip")
    util.download_source(zip_path, URL, download, checksum=CHECKSUM)

    # Extract to a temporary sub-directory and process all seven datasets in one pass.
    extract_dir = os.path.join(path, "_extracted")
    util.unzip(zip_path, extract_dir, remove=True)

    # The archive likely contains a single root folder (e.g. "Datasets/").
    subdirs = [
        d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))
    ]
    data_root = os.path.join(extract_dir, subdirs[0]) if subdirs else extract_dir

    for did in DATASET_CLASSES:
        _create_h5_files(data_root, did, os.path.join(path, f"dataset_{did}"))

    rmtree(extract_dir)

    return out_dir


def get_waenet_paths(
    path: Union[os.PathLike, str],
    dataset_id: int,
    split: Optional[Literal["train", "test"]] = None,
    val_fraction: float = 0.2,
    download: bool = False,
) -> List[str]:
    """Get paths to the WAE-NET HDF5 files.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset_id: Which of the seven sub-datasets to use (1–7).
        split: The data split. Either 'train', 'test', or None for all data.
        val_fraction: Fraction of images reserved for the test split (default 0.2, matching the paper's 8:2 ratio).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the HDF5 files.
    """
    data_dir = get_waenet_data(path, dataset_id, download)
    all_paths = sorted(glob(os.path.join(data_dir, "*.h5")))
    assert len(all_paths) > 0, f"No HDF5 files found in '{data_dir}'"

    if split is None:
        return all_paths

    assert split in ("train", "test"), f"split must be 'train', 'test', or None, got {split!r}"
    n_train = int(len(all_paths) * (1 - val_fraction))
    return all_paths[:n_train] if split == "train" else all_paths[n_train:]


def get_waenet_dataset(
    path: Union[os.PathLike, str],
    dataset_id: int,
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "test"]] = None,
    val_fraction: float = 0.2,
    label_type: Optional[str] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the WAE-NET dataset for multi-class semantic segmentation in electron microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset_id: Which of the seven sub-datasets to use (1–7).
        patch_shape: The patch shape to use for training.
        split: The data split. Either 'train', 'test', or None for all data.
        val_fraction: Fraction of images reserved for the test split (default 0.2).
        label_type: The class to use as segmentation target (e.g. 'cytoplasm', 'nucleus', 'mitochondria').
            If None, defaults to the first non-background class for the given dataset.
            Available classes per dataset are listed in `DATASET_CLASS_NAMES`.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    all_paths = get_waenet_paths(path, dataset_id, split, val_fraction, download)

    if label_type is None:
        label_type = DATASET_CLASS_NAMES[dataset_id][1]

    valid_types = DATASET_CLASS_NAMES[dataset_id]
    if label_type not in valid_types:
        raise ValueError(f"Invalid label_type '{label_type}' for dataset {dataset_id}. Choose from {valid_types}.")

    return torch_em.default_segmentation_dataset(
        raw_paths=all_paths,
        raw_key="raw",
        label_paths=all_paths,
        label_key=f"labels/{label_type}",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_waenet_loader(
    path: Union[os.PathLike, str],
    dataset_id: int,
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: Optional[Literal["train", "test"]] = None,
    val_fraction: float = 0.2,
    label_type: Optional[str] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the WAE-NET dataloader for multi-class semantic segmentation in electron microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        dataset_id: Which of the seven sub-datasets to use (1–7).
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The data split. Either 'train', 'test', or None for all data.
        val_fraction: Fraction of images reserved for the test split (default 0.2).
        label_type: The class to use as segmentation target (e.g. 'cytoplasm', 'nucleus', 'mitochondria').
            If None, defaults to the first non-background class for the given dataset.
            Available classes per dataset are listed in `DATASET_CLASS_NAMES`.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The PyTorch DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_waenet_dataset(path, dataset_id, patch_shape, split, val_fraction, label_type, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
