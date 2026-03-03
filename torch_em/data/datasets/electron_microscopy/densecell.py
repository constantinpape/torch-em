"""The DenseCell dataset contains annotations for semantic segmentation of densely-packed cellular organelles
in serial block-face scanning electron microscopy (SBF-SEM) images of platelet tissue.

The dataset was published in https://doi.org/10.1038/s41598-021-81590-0.
Please cite this publication if you use the dataset in your research.
"""

import os
from shutil import rmtree
from typing import Tuple, Union, Literal, Optional

import numpy as np

import torch_em

from torch.utils.data import Dataset, DataLoader

from .. import util


URL = "https://www.dropbox.com/s/68yclbraqq1diza/platelet_data_1219.zip?dl=1"
CHECKSUM = None

ORGANELLES = {
    1: "cell",
    2: "mitochondrion",
    3: "alpha_granule",
    4: "canalicular_vessel",
    5: "dense_granule",
    6: "dense_core",
}

SPLIT_FILES = {
    "train": {"images": "train-images.tif", "labels": "train-labels.tif"},
    "val": {"images": "eval-images.tif", "labels": "eval-labels.tif"},
    "test": {"images": "test-images.tif", "labels": "test-labels.tif"},
}


def get_densecell_data(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False
) -> str:
    """Download the DenseCell dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to download. Either 'train', 'val', or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath for the downloaded data.
    """
    import h5py
    import tifffile

    data_path = os.path.join(path, f"densecell_{split}.h5")
    if os.path.exists(data_path):
        with h5py.File(data_path, "r") as f:
            if "labels/original" in f:
                return data_path

        # Remove old file with outdated structure.
        os.remove(data_path)

    os.makedirs(path, exist_ok=True)

    # Download and extract the ZIP if the source TIFFs are not available.
    platelet_dir = os.path.join(path, "platelet_data")
    if not os.path.exists(platelet_dir):
        zip_path = os.path.join(path, "platelet_data_1219.zip")
        util.download_source(zip_path, URL, download, checksum=CHECKSUM)
        util.unzip(zip_path, path, remove=True)

    assert os.path.exists(platelet_dir), f"Expected extracted directory at {platelet_dir}"

    for _split, files in SPLIT_FILES.items():
        out_path = os.path.join(path, f"densecell_{_split}.h5")
        if os.path.exists(out_path):
            with h5py.File(out_path, "r") as f:
                if "labels/original" in f:
                    continue

            os.remove(out_path)

        raw = tifffile.imread(os.path.join(platelet_dir, files["images"]))
        labels = tifffile.imread(os.path.join(platelet_dir, files["labels"]))
        assert raw.shape == labels.shape, f"Shape mismatch for {_split}: {raw.shape} vs {labels.shape}"

        labels = labels.astype(np.uint8)
        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/original", data=labels, compression="gzip")
            for label_id, name in ORGANELLES.items():
                # For cells, use all non-background labels to avoid holes from internal organelles.
                if name == "cell":
                    binary_mask = (labels >= 1).astype(np.uint8)
                else:
                    binary_mask = (labels == label_id).astype(np.uint8)

                f.create_dataset(f"labels/{name}", data=binary_mask, compression="gzip")

    rmtree(platelet_dir)

    assert os.path.exists(data_path), data_path
    return data_path


def get_densecell_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False
) -> str:
    """Get paths to the DenseCell data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train', 'val', or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath for the stored data.
    """
    get_densecell_data(path, split, download)
    data_path = os.path.join(path, f"densecell_{split}.h5")
    return data_path


def get_densecell_dataset(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    patch_shape: Tuple[int, int, int],
    label_choice: Optional[str] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get dataset for segmentation of organelles in SBF-SEM platelet images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train', 'val', or 'test'.
        patch_shape: The patch shape to use for training.
        label_choice: The organelle to segment. Available choices are:
            'cell', 'mitochondrion', 'alpha_granule', 'canalicular_vessel', 'dense_granule', 'dense_core'.
            If None, uses 'original' which contains all semantic labels (0-6).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert split in ("train", "val", "test")

    if label_choice is None:
        label_key = "labels/original"
    else:
        valid_choices = list(ORGANELLES.values())
        assert label_choice in valid_choices, f"'{label_choice}' is not valid. Choose from {valid_choices}."
        label_key = f"labels/{label_choice}"

    data_path = get_densecell_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=data_path,
        raw_key="raw",
        label_paths=data_path,
        label_key=label_key,
        patch_shape=patch_shape,
        **kwargs
    )


def get_densecell_loader(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    label_choice: Optional[str] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get dataloader for segmentation of organelles in SBF-SEM platelet images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train', 'val', or 'test'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        label_choice: The organelle to segment. Available choices are:
            'cell', 'mitochondrion', 'alpha_granule', 'canalicular_vessel', 'dense_granule', 'dense_core'.
            If None, uses 'original' which contains all semantic labels (0-6).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The PyTorch DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_densecell_dataset(path, split, patch_shape, label_choice=label_choice, download=download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
