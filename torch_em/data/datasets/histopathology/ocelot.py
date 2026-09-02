"""The OCELOT dataset contains tissue segmentation masks for H&E histopathology images
sourced from TCGA, covering bladder, endometrium, head-and-neck, kidney, prostate, and
stomach cancer.

The dataset is located at https://zenodo.org/records/8417503. The data is licensed under
CC BY-NC 4.0. This dataset is from the publication https://doi.org/10.1109/CVPR52729.2023.02289.
Please cite it in your research.

NOTE: OCELOT also ships point annotations for cell detection, which this module does not expose,
since torch-em only integrates the tissue segmentation masks.
"""

import os
import stat
import zipfile
from glob import glob
from pathlib import PurePosixPath
from typing import List, Literal, Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/8417503/files/ocelot2023_v1.0.1.zip?download=1"
CHECKSUM = "74f46b79e3c4076ca0012d403629ab1e1e412591faf79a910d4d8bdd92c47920"
SPLITS = ("train", "val", "test")


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
            or first_part != "ocelot2023_v1.0.1"
        ):
            raise RuntimeError(f"Unsafe archive member: {member.filename}")

        file_type = stat.S_IFMT(member.external_attr >> 16)
        if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise RuntimeError(f"Unsupported archive member type: {member.filename}")
        if not member.is_dir():
            file_members.append(member)

    extracted_size = sum(member.file_size for member in file_members)
    archive_size = os.path.getsize(zip_path)
    if len(file_members) > 20000 or extracted_size > 3_000_000_000 or extracted_size > 5 * archive_size:
        raise RuntimeError("The OCELOT archive exceeds the expected extraction limits.")


def _has_data(data_dir):
    return all(
        os.path.isdir(os.path.join(data_dir, "images", split, "tissue"))
        and os.path.isdir(os.path.join(data_dir, "annotations", split, "tissue"))
        for split in SPLITS
    )


def get_ocelot_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the OCELOT dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the data is stored.
    """
    data_dir = os.path.join(path, "ocelot2023_v1.0.1")
    if _has_data(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "ocelot2023_v1.0.1.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    _validate_archive(zip_path)
    util.unzip(zip_path=zip_path, dst=path)

    if not _has_data(data_dir):
        raise RuntimeError("The OCELOT archive does not contain the expected image and mask folders.")
    return data_dir


def get_ocelot_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "val", "test"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the OCELOT tissue images and semantic segmentation masks.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use. By default all splits ('train', 'val', 'test') are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_ocelot_data(path, download)
    splits = SPLITS if split is None else (split,)

    raw_paths, label_paths = [], []
    for this_split in splits:
        split_raw_paths = sorted(glob(os.path.join(data_dir, "images", this_split, "tissue", "*.jpg")))
        split_label_paths = sorted(glob(os.path.join(data_dir, "annotations", this_split, "tissue", "*.png")))
        if not split_raw_paths or len(split_raw_paths) != len(split_label_paths):
            raise RuntimeError(f"Invalid OCELOT raw-label pairing for split '{this_split}'.")
        if any(
            os.path.splitext(os.path.basename(raw_path))[0] != os.path.splitext(os.path.basename(label_path))[0]
            for raw_path, label_path in zip(split_raw_paths, split_label_paths)
        ):
            raise RuntimeError(f"Mismatched OCELOT raw-label names for split '{this_split}'.")
        raw_paths.extend(split_raw_paths)
        label_paths.extend(split_label_paths)

    return raw_paths, label_paths


def get_ocelot_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "val", "test"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the OCELOT dataset for tissue segmentation.

    The masks use label 1 for background, 2 for cancer area, and 255 for unlabeled pixels.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The split to use. By default all splits ('train', 'val', 'test') are used.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_ocelot_paths(path, split, download)

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


def get_ocelot_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "val", "test"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the OCELOT dataloader for tissue segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The split to use. By default all splits ('train', 'val', 'test') are used.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ocelot_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
