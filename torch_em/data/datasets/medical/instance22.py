"""The INSTANCE dataset contains annotations for intracranial hemorrhage segmentation in
non-contrast head CT (NCCT) scans.

It comprises the training set of the INSTANCE2022 challenge (https://instance.grand-challenge.org):
100 NCCT volumes with a voxel-wise annotation of the intracranial hemorrhage, refined by 10 radiologists.
The 30 validation volumes are distributed without annotations and are therefore not included here.

NOTE: The label legend is as follows:
- background: 0, intracranial hemorrhage: 1

NOTE: The organizers announced on 2026/09/11 that they have made the dataset open to the public, and state
that the download link can be found in the dataset part of the challenge page. That page,
https://instance.grand-challenge.org/Dataset/, still returns HTTP 403 to users who are not signed in to
grand-challenge.org, so the data cannot be downloaded automatically. There is no mirror of this data
elsewhere. Please follow these steps:
- Register at https://instance.grand-challenge.org/ and join the challenge. If the dataset page is still
  not accessible to you, write to INSTANCE2022@outlook.com and, if you are participating in the challenge,
  send the signed data agreement
  (https://github.com/PerceptionComputingLab/INSTANCE2022/blob/main/Agreements/instance2022_agreements.pdf).
- Follow the download link on https://instance.grand-challenge.org/Dataset/ and extract the archive into
  '<path>', such that '<path>/train_2/data/001.nii.gz' and '<path>/train_2/label/001.nii.gz' exist
  (a folder named 'train' instead of 'train_2' is also accepted).
  The case ids are the zero-padded numbers 001 - 100.

The dataset is located at https://instance.grand-challenge.org/Dataset/. The annotations are released
under a CC BY-NC-ND license and https://instance.grand-challenge.org/Participation/ states that any other
use of the data, including redistribution, is not allowed, so please make sure that you are allowed to use
the data for your purpose.

This dataset is from the publication https://doi.org/10.48550/arXiv.2301.03281.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


LABEL_IDS = {"background": 0, "hemorrhage": 1}


def get_instance22_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Obtain the INSTANCE dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    # The archive is either extracted directly into 'path' or into a folder named after the archive.
    # NOTE: The official archive is called 'train_2.zip' and extracts to a folder 'train_2', while the dataset
    # description calls it 'train', so both spellings are accepted. The folder is identified by its subfolders.
    candidates = natsorted(glob(os.path.join(path, "**", "train*"), recursive=True))
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "data")) and os.path.isdir(os.path.join(candidate, "label")):
            return candidate

    msg = f"It's expected to place the extracted INSTANCE2022 training data at '{path}'. "
    msg += "'torch_em' cannot download this dataset, as it is only accessible to users who are signed in to "
    msg += "grand-challenge.org. See 'torch_em.data.datasets.medical.instance22' for the manual download "
    msg += "instructions."
    if download:
        raise NotImplementedError(msg)
    else:
        raise FileNotFoundError(msg)


def get_instance22_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the INSTANCE data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_instance22_data(path, download)

    # NOTE: The label paths are built from the data directory instead of replacing 'data' in the image paths,
    # because the path to the dataset itself may contain a folder called 'data'.
    raw_paths = natsorted(glob(os.path.join(data_dir, "data", "*.nii.gz")))
    label_paths = [os.path.join(data_dir, "label", os.path.basename(p)) for p in raw_paths]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_instance22_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the INSTANCE dataset for intracranial hemorrhage segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_instance22_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_instance22_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the INSTANCE dataloader for intracranial hemorrhage segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_instance22_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
