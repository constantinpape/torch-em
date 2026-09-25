"""The TNBC-Mito dataset contains annotations for semantic segmentation of mitochondria in transmission
electron microscopy (TEM) images of triple-negative breast cancer (TNBC) models and Drp1-deficient mouse
skeletal muscle.

The data covers four cohorts: DRP1-KO (Drp1-deficient mouse primary skeletal muscle cells), HCI-010 and
PIM001-P (TNBC patient-derived xenograft models), and Mixture (a pool of TNBC cell-line and xenograft images).

The dataset is available at https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2271 under the CC0 license.
The dataset was published in https://doi.org/10.1101/2025.02.19.635300.
Please cite this publication if you use the dataset in your research.
"""

import os
from glob import glob
from typing import List, Literal, Optional, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


BASE_URL = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/271/S-BIAD2271/Files"
MANIFEST_URL = f"{BASE_URL}/tem-seg-data_mitochondria_masks.tsv"

COHORTS = ("DRP1-KO", "HCI-010", "Mixture", "PIM001-P")
SPLIT_DIRS = {"tra_val": "tra_val", "tst": "tst"}


def _read_manifest(manifest_path):
    """Parse the mask-to-source-image manifest into (mask_path, image_path) tuples."""
    pairs = []
    with open(manifest_path) as f:
        next(f)  # Skip the header line.
        for line in f:
            mask_path, image_path = line.strip().split("\t")
            pairs.append((mask_path, image_path))
    return pairs


def get_tnbc_mito_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the TNBC-Mito dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder that mirrors the remote 'tem-seg/data' directory structure.
    """
    data_root = os.path.join(path, "tem-seg", "data")
    marker = os.path.join(path, ".download_complete")
    if os.path.exists(marker):
        return data_root

    os.makedirs(path, exist_ok=True)

    manifest_path = os.path.join(path, "tem-seg-data_mitochondria_masks.tsv")
    util.download_source(manifest_path, MANIFEST_URL, download, checksum=None)
    pairs = _read_manifest(manifest_path)

    for mask_rel, image_rel in pairs:
        mask_path = os.path.join(path, mask_rel)
        image_path = os.path.join(path, image_rel)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        util.download_source(mask_path, f"{BASE_URL}/{mask_rel}", download, checksum=None)
        util.download_source(image_path, f"{BASE_URL}/{image_rel}", download, checksum=None)
        _normalize_mask(mask_path)

    with open(marker, "w"):
        pass

    return data_root


def _normalize_mask(mask_path):
    """Re-save a palette-indexed mask PNG as a single-channel TIFF.

    The source masks are palette ('P' mode) PNGs. Generic image readers expand palette images to RGB,
    which breaks single-channel label loading, so this writes the raw index values (already 0/1) once
    as a plain grayscale TIFF next to the original file.
    """
    import numpy as np
    import tifffile
    from PIL import Image

    normalized_path = f"{os.path.splitext(mask_path)[0]}.tif"
    if os.path.exists(normalized_path):
        return
    labels = np.array(Image.open(mask_path))
    tifffile.imwrite(normalized_path, labels.astype("uint8"))


def get_tnbc_mito_paths(
    path: Union[os.PathLike, str],
    split: Literal["tra_val", "tst"],
    cohort: Optional[str] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the TNBC-Mito raw images and mitochondria masks.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'tra_val' (pooled train / validation) or 'tst' (test).
        cohort: The cohort to restrict to. One of 'DRP1-KO', 'HCI-010', 'Mixture', 'PIM001-P'.
            If None, uses all cohorts that provide the requested split.
        download: Whether to download the data if it is not present.

    Returns:
        The list of raw image paths.
        The list of mitochondria mask paths.
    """
    assert split in SPLIT_DIRS, f"split must be one of {list(SPLIT_DIRS)}, got {split!r}"
    if cohort is not None:
        assert cohort in COHORTS, f"cohort must be one of {COHORTS}, got {cohort!r}"

    data_root = get_tnbc_mito_data(path, download)

    cohorts = [cohort] if cohort is not None else list(COHORTS)
    raw_paths, label_paths = [], []
    for c in cohorts:
        image_dir = os.path.join(data_root, c, split, "slide_images")
        mask_dir = os.path.join(data_root, c, split, "mitochondria", "masks")
        if not os.path.exists(image_dir):
            continue
        images = sorted(glob(os.path.join(image_dir, "*.tif")))
        for image_path in images:
            fname = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(mask_dir, f"{fname}.tif")
            assert os.path.exists(mask_path), f"Missing normalized mask for '{image_path}' at '{mask_path}'"
            raw_paths.append(image_path)
            label_paths.append(mask_path)

    assert len(raw_paths) > 0, f"No images found for split '{split}' (cohort={cohort}) in '{data_root}'"
    return raw_paths, label_paths


def get_tnbc_mito_dataset(
    path: Union[os.PathLike, str],
    split: Literal["tra_val", "tst"],
    patch_shape: Tuple[int, int],
    cohort: Optional[str] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the TNBC-Mito dataset for mitochondria segmentation in TEM images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'tra_val' (pooled train / validation) or 'tst' (test).
        patch_shape: The patch shape to use for training.
        cohort: The cohort to restrict to. One of 'DRP1-KO', 'HCI-010', 'Mixture', 'PIM001-P'.
            If None, uses all cohorts that provide the requested split.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_tnbc_mito_paths(path, split, cohort, download)
    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs,
    )


def get_tnbc_mito_loader(
    path: Union[os.PathLike, str],
    split: Literal["tra_val", "tst"],
    patch_shape: Tuple[int, int],
    batch_size: int,
    cohort: Optional[str] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the TNBC-Mito dataloader for mitochondria segmentation in TEM images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'tra_val' (pooled train / validation) or 'tst' (test).
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        cohort: The cohort to restrict to. One of 'DRP1-KO', 'HCI-010', 'Mixture', 'PIM001-P'.
            If None, uses all cohorts that provide the requested split.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch
            DataLoader.

    Returns:
        The PyTorch DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_tnbc_mito_dataset(path, split, patch_shape, cohort=cohort, download=download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
