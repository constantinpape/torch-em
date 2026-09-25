"""The TIGER WSIROIS dataset contains semantic tissue masks for H&E breast cancer histopathology images.

The dataset is located at https://doi.org/10.5281/zenodo.6014422. The annotations and RUMC/JB images are
licensed under CC BY-NC 4.0, while TCGA-derived images retain their original TCGA-BRCA rights. This dataset
is from the publication https://doi.org/10.1038/s41467-026-72956-x. Please cite it in your research.
"""

import os
import stat
import zipfile
from glob import glob
from pathlib import PurePosixPath
from typing import List, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/6014422/files/roi-level-annotations.zip/content"
CHECKSUM = "94bf1a00a61b8d264a6d8d9f213000617766ce65823ab33af86498041bf866dd"
SUBSETS = ("tissue-bcss", "tissue-cells")


def _validate_archive(zip_path):
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = archive.infolist()

    file_members = []
    for member in members:
        member_path = PurePosixPath(member.filename)
        first_part = member_path.parts[0] if member_path.parts else ""
        if (
            not member_path.parts
            or member_path.is_absolute()
            or ".." in member_path.parts
            or "\\" in member.filename
            or ":" in first_part
            or first_part != "roi-level-annotations"
        ):
            raise RuntimeError(f"Unsafe archive member: {member.filename}")

        file_type = stat.S_IFMT(member.external_attr >> 16)
        if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise RuntimeError(f"Unsupported archive member type: {member.filename}")
        if not member.is_dir():
            file_members.append(member)

    extracted_size = sum(member.file_size for member in file_members)
    archive_size = os.path.getsize(zip_path)
    if len(file_members) > 5000 or extracted_size > 3_000_000_000 or extracted_size > 2 * archive_size:
        raise RuntimeError("The TIGER archive exceeds the expected extraction limits.")


def _has_data(data_dir):
    return all(
        os.path.isdir(os.path.join(data_dir, subset, folder))
        for subset in SUBSETS
        for folder in ("images", "masks")
    )


def get_tiger_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the TIGER WSIROIS ROI-level dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the ROI-level annotations.
    """
    data_dir = os.path.join(path, "roi-level-annotations")
    if _has_data(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "roi-level-annotations.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    _validate_archive(zip_path)
    util.unzip(zip_path=zip_path, dst=path)

    if not _has_data(data_dir):
        raise RuntimeError("The TIGER archive does not contain the expected image and mask folders.")
    return data_dir


def get_tiger_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the TIGER WSIROIS ROI images and semantic tissue masks.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_tiger_data(path, download)
    raw_paths, label_paths = [], []

    for subset in SUBSETS:
        subset_raw_paths = sorted(glob(os.path.join(data_dir, subset, "images", "*.png")))
        subset_label_paths = sorted(glob(os.path.join(data_dir, subset, "masks", "*.png")))
        if not subset_raw_paths or len(subset_raw_paths) != len(subset_label_paths):
            raise RuntimeError(f"Invalid TIGER raw-label pairing for subset '{subset}'.")
        if any(
            os.path.basename(raw_path) != os.path.basename(label_path)
            for raw_path, label_path in zip(subset_raw_paths, subset_label_paths)
        ):
            raise RuntimeError(f"Mismatched TIGER raw-label names for subset '{subset}'.")
        raw_paths.extend(subset_raw_paths)
        label_paths.extend(subset_label_paths)

    return raw_paths, label_paths


def get_tiger_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the TIGER dataset for semantic breast tissue segmentation.

    The masks use label 0 for excluded pixels and labels 1 through 7 for invasive tumor,
    tumor-associated stroma, in-situ tumor, healthy glands, necrosis, inflamed stroma, and rest.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_tiger_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        ndim=2,
        with_channels=True,
        **kwargs,
    )


def get_tiger_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the TIGER dataloader for semantic breast tissue segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_tiger_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
