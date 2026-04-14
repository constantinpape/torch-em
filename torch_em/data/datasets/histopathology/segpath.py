"""SegPath contains semantic segmentation masks for H&E stained histopathology images from diverse cancer tissues.

The dataset is located at https://dakomura.github.io/SegPath/ (across several Zenodo links).
The dataset is from the publication https://doi.org/10.1016/j.patter.2023.100688.
Please cite it if you use this dataset for your research.
"""

import csv
import gzip
import os
import tarfile
from glob import glob
from typing import List, Literal, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "epithelium": {
        "data": "https://zenodo.org/api/records/7412731/files/panCK_Epithelium.tar.gz/content",
        "metadata": "https://zenodo.org/api/records/7412731/files/panCK_fileinfo.csv/content",
        "data_name": "panCK_Epithelium.tar.gz",
        "metadata_name": "panCK_fileinfo.csv",
    },
    "smooth_muscle": {
        "data": "https://zenodo.org/api/records/7412732/files/aSMA_SmoothMuscle.tar.gz/content",
        "metadata": "https://zenodo.org/api/records/7412732/files/aSMA_fileinfo.csv/content",
        "data_name": "aSMA_SmoothMuscle.tar.gz",
        "metadata_name": "aSMA_fileinfo.csv",
    },
    "red_blood_cells": {
        "data": "https://zenodo.org/api/records/7412580/files/CD235a_RBC.tar.gz/content",
        "metadata": "https://zenodo.org/api/records/7412580/files/CD235a_fileinfo.csv/content",
        "data_name": "CD235a_RBC.tar.gz",
        "metadata_name": "CD235a_fileinfo.csv",
    },
    "leukocytes": {
        "data": "https://zenodo.org/api/records/7412739/files/CD45RB_Leukocyte.tar.gz/content",
        "metadata": "https://zenodo.org/api/records/7412739/files/CD45RB_fileinfo.csv/content",
        "data_name": "CD45RB_Leukocyte.tar.gz",
        "metadata_name": "CD45RB_fileinfo.csv",
    },
    "lymphocytes": {
        "data": "https://zenodo.org/api/records/7412529/files/CD3CD20_Lymphocyte.tar.gz/content",
        "metadata": "https://zenodo.org/api/records/7412529/files/CD3CD20_fileinfo.csv/content",
        "data_name": "CD3CD20_Lymphocyte.tar.gz",
        "metadata_name": "CD3CD20_fileinfo.csv",
    },
    "endothelium": {
        "data": "https://zenodo.org/api/records/7412512/files/ERG_Endothelium.tar.gz/content",
        "metadata": "https://zenodo.org/api/records/7412512/files/ERG_fileinfo.csv/content",
        "data_name": "ERG_Endothelium.tar.gz",
        "metadata_name": "ERG_fileinfo.csv",
    },
    "plasma_cells": {
        "data": "https://zenodo.org/api/records/7412500/files/MIST1_PlasmaCell.tar.gz/content",
        "metadata": "https://zenodo.org/api/records/7412500/files/MIST1_fileinfo.csv/content",
        "data_name": "MIST1_PlasmaCell.tar.gz",
        "metadata_name": "MIST1_fileinfo.csv",
    },
    "myeloid_cells": {
        "data": "https://zenodo.org/api/records/7412690/files/MNDA_MyeloidCell.tar.gz/content",
        "metadata": "https://zenodo.org/api/records/7412690/files/MNDA_fileinfo.csv/content",
        "data_name": "MNDA_MyeloidCell.tar.gz",
        "metadata_name": "MNDA_fileinfo.csv",
    },
}


def _to_cell_types(cell_types):
    if cell_types is None:
        return list(URLS)
    if isinstance(cell_types, str):
        cell_types = [cell_types]
    invalid_cell_types = set(cell_types) - set(URLS)
    if invalid_cell_types:
        raise ValueError(
            f"Invalid cell type choices: {sorted(invalid_cell_types)}. Choose from {sorted(URLS)}."
        )
    return cell_types


def _is_gzip(path):
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def _extract_data(path):
    data_folder = os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0]
    extract_path = os.path.join(os.path.dirname(path), data_folder)
    if os.path.exists(extract_path):
        return

    extract_root = os.path.dirname(path)
    with tarfile.open(path) as f:
        for member in f.getmembers():
            member_path = os.path.abspath(os.path.join(extract_root, member.name))
            if os.path.commonpath([os.path.abspath(extract_root), member_path]) != os.path.abspath(extract_root):
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
        f.extractall(extract_root)


def get_segpath_data(
    path: Union[os.PathLike, str],
    cell_types: Optional[Union[str, List[str]]] = None,
    download: bool = False,
) -> None:
    """Download the SegPath data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        cell_types: The cell types to download. By default all cell types are downloaded.
        download: Whether to download the data if it is not present.
    """
    os.makedirs(path, exist_ok=True)
    if not download:
        return

    for cell_type in _to_cell_types(cell_types):
        source = URLS[cell_type]
        data_path = os.path.join(path, source["data_name"])
        metadata_path = os.path.join(path, source["metadata_name"])
        data_folder = os.path.splitext(os.path.splitext(source["data_name"])[0])[0]
        extracted_path = os.path.join(path, data_folder)

        util.download_source(metadata_path, source["metadata"], download, checksum=None)

        if not os.path.exists(extracted_path):
            util.download_source(data_path, source["data"], download, checksum=None)
            _extract_data(data_path)


def _get_paths_from_metadata(path, cell_type, split):
    source = URLS[cell_type]
    metadata_path = os.path.join(path, source["metadata_name"])
    image_paths, label_paths = [], []

    open_file = gzip.open if _is_gzip(metadata_path) else open
    with open_file(metadata_path, mode="rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split is not None and row["train_val_test"] != split:
                continue

            filename = row["filename"]
            if not filename.endswith("_HE.png"):
                continue

            image_path = os.path.join(path, filename)
            label_path = os.path.join(path, filename.replace("_HE.png", "_mask.png"))
            if not os.path.exists(image_path) or not os.path.exists(label_path):
                continue

            image_paths.append(image_path)
            label_paths.append(label_path)

    return image_paths, label_paths


def _get_paths_from_files(path, cell_type, split):
    if split is not None:
        raise RuntimeError(
            "The SegPath metadata CSV is required for split selection, but it could not be found. "
            "Please download the metadata with `download=True` or place it into the dataset folder."
        )

    data_name = os.path.splitext(os.path.splitext(URLS[cell_type]["data_name"])[0])[0]
    image_paths = sorted(glob(os.path.join(path, data_name, "*_HE.png")))
    label_paths = [image_path.replace("_HE.png", "_mask.png") for image_path in image_paths]
    paired_paths = [
        (image_path, label_path) for image_path, label_path in zip(image_paths, label_paths)
        if os.path.exists(label_path)
    ]
    if not paired_paths:
        return [], []

    image_paths, label_paths = zip(*paired_paths)
    return list(image_paths), list(label_paths)


def get_segpath_paths(
    path: Union[os.PathLike, str],
    cell_types: Optional[Union[str, List[str]]] = None,
    split: Optional[Literal["train", "val", "test"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the SegPath data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        cell_types: The cell types to use. By default all cell types are used.
        split: The split to use. Either "train", "val", "test" or None for all images.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split is not None and split not in ("train", "val", "test"):
        raise ValueError(f"'{split}' is not a valid split choice.")

    cell_types = _to_cell_types(cell_types)
    get_segpath_data(path, cell_types, download)

    image_paths, label_paths = [], []
    for cell_type in cell_types:
        metadata_path = os.path.join(path, URLS[cell_type]["metadata_name"])
        if os.path.exists(metadata_path):
            this_image_paths, this_label_paths = _get_paths_from_metadata(path, cell_type, split)
        else:
            this_image_paths, this_label_paths = _get_paths_from_files(path, cell_type, split)

        image_paths.extend(this_image_paths)
        label_paths.extend(this_label_paths)

    if not image_paths:
        raise RuntimeError("Could not find any SegPath images and masks for the requested settings.")

    return image_paths, label_paths


def get_segpath_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    cell_types: Optional[Union[str, List[str]]] = None,
    split: Optional[Literal["train", "val", "test"]] = None,
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> Dataset:
    """Get the SegPath dataset for semantic segmentation in H&E stained histopathology images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        cell_types: The cell types to use. By default all cell types are used.
        split: The split to use. Either "train", "val", "test" or None for all images.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_segpath_paths(path, cell_types, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        label_dtype=label_dtype,
        is_seg_dataset=False,
        **kwargs
    )


def get_segpath_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    cell_types: Optional[Union[str, List[str]]] = None,
    split: Optional[Literal["train", "val", "test"]] = None,
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SegPath dataloader.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        cell_types: The cell types to use. By default all cell types are used.
        split: The split to use. Either "train", "val", "test" or None for all images.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_segpath_dataset(
        path=path, patch_shape=patch_shape, cell_types=cell_types, split=split, download=download,
        label_dtype=label_dtype, resize_inputs=resize_inputs, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
