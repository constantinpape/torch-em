"""The Lewy Mito dataset contains annotations for mitochondria segmentation in transmission electron microscopy (TEM)
images of post-mortem human brain tissue.

The tissue is from the substantia nigra (SN), entorhinal cortex (ENT) and hippocampus (HC) of donors with Parkinson's
disease (PD) and dementia with Lewy bodies (DLB), and of healthy controls. The images were acquired as part of a
correlative light and electron microscopy workflow. The data is organized into the following groups:
- 'ENT-DLB', 'ENT-PD', 'HC-ENT', 'HC-SN', 'SN-DLB', 'SN-PD': Images from the regions and conditions in their names.
- 'HN-ENT-DLB', 'HN-ENT-PD', 'HN-SN-DLB', 'HN-SN-PD': The same regions and conditions, from a second image set
  (called 'HN' in the original data, its meaning is not explained in the dataset description).

There are roughly 16,500 small image crops (a few hundred pixels per side) with manual (and proofread) mitochondria
masks. In total the download size is ~7 GB. The files are only available individually (not as an archive),
so downloading the full dataset takes some time. Use `groups` to download only a subset.

The data is available at https://www.ebi.ac.uk/biostudies/studies/S-BIAD3729 (doi: 10.6019/S-BIAD3729).
The licence of the BioStudies entry could not be confirmed (BioImage Archive entries are CC0 by default).
The dataset is from the publication "Correlative Ultrastructural Mapping of Lewy Pathology Reveals Regional Diversity
in Parkinson's and Dementia with Lewy bodies". Please cite it if you use this dataset in your research.
"""

import os
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/729/S-BIAD3729/Files"
FILE_LIST_URL = "https://www.ebi.ac.uk/biostudies/files/S-BIAD3729/Shafiei_Mito-Segmentation.json"
ROOT_FOLDER = "Shafiei_Mito-Segmentation"
GROUPS = [
    "ENT-DLB", "ENT-PD", "HC-ENT", "HC-SN", "HN-ENT-DLB", "HN-ENT-PD", "HN-SN-DLB", "HN-SN-PD", "SN-DLB", "SN-PD",
]
# NOTE: The file list of the study does not contain checksums, so the downloaded files are not verified.


def _get_file_list(path):
    list_path = os.path.join(path, "file_list.json")
    util.download_source(list_path, FILE_LIST_URL, download=True)
    with open(list_path) as f:
        return json.load(f)


def _download_group(path, group, file_list, num_workers):
    # Find all image-mask pairs of the group. Empty (broken) files and their partners are skipped.
    sizes = {x["path"]: x["size"] for x in file_list if x["path"].endswith(".tif")}
    pairs = []
    for rel_path, size in sizes.items():
        parts = rel_path.split("/")
        if parts[1] != group or parts[3] != "crops":
            continue
        mask_rel_path = "/".join(parts[:3] + ["masks", parts[-1]])
        if size == 0 or sizes.get(mask_rel_path, 0) == 0:
            continue
        pairs.append((rel_path, mask_rel_path))

    def _is_complete(rel_path):
        file_path = os.path.join(path, rel_path)
        return os.path.exists(file_path) and os.path.getsize(file_path) == sizes[rel_path]

    def _download(rel_path):
        # The server sometimes returns an html page (instead of the file) under load.
        # We check the downloaded file against the size in the file list and retry if necessary.
        file_path = os.path.join(path, rel_path)
        for attempt in range(5):
            try:
                if os.path.exists(file_path) and not _is_complete(rel_path):
                    os.remove(file_path)
                util.download_source(file_path, f"{BASE_URL}/{rel_path}", download=True)
                if _is_complete(rel_path):
                    return
            except Exception:
                pass
            time.sleep(2 ** attempt)
        return rel_path

    todo = [p for pair in pairs for p in pair if not _is_complete(p)]
    for rel_path in todo:
        os.makedirs(os.path.dirname(os.path.join(path, rel_path)), exist_ok=True)
    with ThreadPoolExecutor(num_workers) as pool:
        failed = {rel_path for rel_path in pool.map(_download, todo) if rel_path is not None}

    # Some files that are listed in the study cannot be retrieved from the server (it returns an index page).
    # We skip the image-mask pairs that are affected by this.
    if failed:
        warnings.warn(f"Skipping {len(failed)} files of group '{group}' that cannot be downloaded: {sorted(failed)}")
        for rel_path in failed:
            if os.path.exists(os.path.join(path, rel_path)):
                os.remove(os.path.join(path, rel_path))
        pairs = [pair for pair in pairs if not failed.intersection(pair)]

    return pairs


def get_lewy_mito_data(
    path: Union[os.PathLike, str],
    groups: Optional[List[str]] = None,
    download: bool = False,
    num_workers: int = 8,
) -> List[Tuple[str, str]]:
    """Download the Lewy Mito dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        groups: The groups (region and condition) to download. By default all groups are used.
            Choose from 'ENT-DLB', 'ENT-PD', 'HC-ENT', 'HC-SN', 'HN-ENT-DLB', 'HN-ENT-PD', 'HN-SN-DLB',
            'HN-SN-PD', 'SN-DLB' and 'SN-PD'.
        download: Whether to download the data if it is not present.
        num_workers: The number of threads used for downloading the individual files.

    Returns:
        The relative filepaths (image, mask) of all image-mask pairs in the selected groups.
    """
    groups = GROUPS if groups is None else groups
    for group in groups:
        if group not in GROUPS:
            raise ValueError(f"'{group}' is not a valid group. Please choose from {GROUPS}.")

    os.makedirs(path, exist_ok=True)

    # Cache the pair list per group, so that the data can be used offline once it was downloaded.
    pairs = []
    file_list = None
    for group in groups:
        pair_path = os.path.join(path, f"pairs_{group}.json")
        if os.path.exists(pair_path):
            with open(pair_path) as f:
                group_pairs = json.load(f)
        else:
            if not download:
                raise RuntimeError(f"Cannot find the data for group '{group}' at {path}. Set download=True.")
            file_list = _get_file_list(path) if file_list is None else file_list
            group_pairs = _download_group(path, group, file_list, num_workers)
            with open(pair_path, "w") as f:
                json.dump(group_pairs, f)
        pairs.extend(group_pairs)

    return pairs


def get_lewy_mito_paths(
    path: Union[os.PathLike, str],
    groups: Optional[List[str]] = None,
    download: bool = False,
    num_workers: int = 8,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Lewy Mito data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        groups: The groups (region and condition) to use. By default all groups are used.
        download: Whether to download the data if it is not present.
        num_workers: The number of threads used for downloading the individual files.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    pairs = get_lewy_mito_data(path, groups, download, num_workers)
    pairs = natsorted(pairs)
    raw_paths = [os.path.join(path, raw) for raw, _ in pairs]
    label_paths = [os.path.join(path, label) for _, label in pairs]
    return raw_paths, label_paths


def _binary_labels(labels):
    return (labels > 0).astype("uint8")


def get_lewy_mito_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    groups: Optional[List[str]] = None,
    binary: bool = True,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Lewy Mito dataset for mitochondria segmentation in TEM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        groups: The groups (region and condition) to use. By default all groups are used.
        binary: Whether to return a binary foreground mask. Otherwise the mask values are returned as they are.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_lewy_mito_paths(path, groups, download)

    if binary:
        kwargs = util.update_kwargs(kwargs, "label_transform", _binary_labels)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_lewy_mito_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    groups: Optional[List[str]] = None,
    binary: bool = True,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Lewy Mito dataloader for mitochondria segmentation in TEM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        groups: The groups (region and condition) to use. By default all groups are used.
        binary: Whether to return a binary foreground mask. Otherwise the mask values are returned as they are.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lewy_mito_dataset(path, patch_shape, groups, binary, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
