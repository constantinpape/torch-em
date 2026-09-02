"""This dataset contains annotations for 3d instance segmentation of oocytes in whole-mount
immunofluorescence imaging of the mouse ovary, at adult and perinatal developmental timepoints.

The dataset is located at https://doi.org/10.5061/dryad.nk98sf81r.
This dataset is from the publication https://doi.org/10.1093/biolre/ioaf023.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def get_oocount_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Instructions to obtain the OoCount training data.

    NOTE: Dryad blocks automated downloads for this dataset. Please manually download
    'TrainingDatasets.zip' from https://datadryad.org/dataset/doi:10.5061/dryad.nk98sf81r
    and place it at the given 'path'.

    Args:
        path: Filepath to a folder where the manually downloaded zip file is placed.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data directory.
    """
    data_dir = os.path.join(path, "TrainingDatasets")
    if os.path.exists(data_dir):
        return data_dir

    if download:
        raise NotImplementedError(
            "The OoCount dataset cannot be downloaded automatically because Dryad blocks automated requests. "
            "Please manually download 'TrainingDatasets.zip' from "
            "https://datadryad.org/dataset/doi:10.5061/dryad.nk98sf81r and place it at the given 'path'."
        )

    zip_path = os.path.join(path, "TrainingDatasets.zip")
    if not os.path.exists(zip_path):
        raise RuntimeError(
            f"The manually downloaded zip file should be placed at '{zip_path}'. Please download "
            "'TrainingDatasets.zip' from https://datadryad.org/dataset/doi:10.5061/dryad.nk98sf81r."
        )

    util.unzip(zip_path=zip_path, dst=path, remove=False)
    assert os.path.exists(data_dir), data_dir

    return data_dir


def get_oocount_paths(
    path: Union[os.PathLike, str],
    timepoint: Literal["adult", "perinatal"] = "adult",
    split: Literal["train", "val"] = "train",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the OoCount data.

    Args:
        path: Filepath to a folder where the data is stored.
        timepoint: The ovary developmental timepoint. Either 'adult' or 'perinatal'.
        split: The data split to use. 'train' uses the training images, 'val' uses the held-out QC images.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if timepoint not in ("adult", "perinatal"):
        raise ValueError(f"'{timepoint}' is not a valid timepoint. Choose 'adult' or 'perinatal'.")
    if split not in ("train", "val"):
        raise ValueError(f"'{split}' is not a valid split. Choose 'train' or 'val'.")

    data_dir = get_oocount_data(path, download)

    sample_dir = os.path.join(data_dir, f"Vasa-{timepoint.capitalize()}")
    image_dir = os.path.join(sample_dir, "Images" if split == "train" else "QC Images")
    label_dir = os.path.join(sample_dir, "Masks" if split == "train" else "QC Masks")

    raw_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
    label_paths = natsorted(glob(os.path.join(label_dir, "*.tif")))

    if len(raw_paths) == 0:
        raise RuntimeError(f"No image files found in {image_dir}.")
    if len(raw_paths) != len(label_paths):
        raise RuntimeError(
            f"Mismatch between images ({len(raw_paths)}) and masks ({len(label_paths)}) in {sample_dir}."
        )

    return raw_paths, label_paths


def get_oocount_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    timepoint: Literal["adult", "perinatal"] = "adult",
    split: Literal["train", "val"] = "train",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the OoCount dataset for oocyte instance segmentation in 3d fluorescence microscopy.

    Args:
        path: Filepath to a folder where the data is stored.
        patch_shape: The patch shape to use for training.
        timepoint: The ovary developmental timepoint. Either 'adult' or 'perinatal'.
        split: The data split to use. 'train' uses the training images, 'val' uses the held-out QC images.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_oocount_paths(path, timepoint, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_oocount_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    timepoint: Literal["adult", "perinatal"] = "adult",
    split: Literal["train", "val"] = "train",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the OoCount dataloader for oocyte instance segmentation in 3d fluorescence microscopy.

    Args:
        path: Filepath to a folder where the data is stored.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        timepoint: The ovary developmental timepoint. Either 'adult' or 'perinatal'.
        split: The data split to use. 'train' uses the training images, 'val' uses the held-out QC images.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_oocount_dataset(path, patch_shape, timepoint, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
