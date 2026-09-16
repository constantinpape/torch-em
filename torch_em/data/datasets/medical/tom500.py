"""The TOM500 dataset contains annotations for multi-organ orbital segmentation
in T2-weighted orbital MRI for thyroid eye disease.

The dataset consists of 500 patients with thyroid eye disease, each with a coronal T2-weighted MRI scan
(512 x 512 x 20 voxels) and corresponding segmentations of nine orbital structures: the optic nerve,
orbital fat, lacrimal gland, eyeball and five extraocular muscles (superior rectus and levator palpebrae
superioris complex, inferior rectus, medial rectus, lateral rectus, superior oblique). The annotations
were created by three junior annotators and reviewed by an expert radiologist. The data is split into
400 training and 100 validation scans.

The dataset is located at https://doi.org/10.6084/m9.figshare.27133389 and is distributed under the
CC0 1.0 license.

This dataset is from the publication https://doi.org/10.1038/s41597-025-04427-9.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/49499655"
CHECKSUM = "80bbd9934584573aabef3525ab1736d1c3ed7d9dcc9e8cdff28b3d22627e089f"


def get_tom500_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the TOM500 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "TOM500")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "TOM500.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_tom500_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'val'], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the TOM500 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_tom500_data(path, download)

    if split not in ("train", "val"):
        raise ValueError(f"'{split}' is not a valid split. Choose either 'train' or 'val'.")

    raw_paths = natsorted(glob(os.path.join(data_dir, split, "image", "*.nii.gz")))
    label_paths = natsorted(glob(os.path.join(data_dir, split, "label", "*.nii.gz")))
    assert len(raw_paths) > 0 and len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def get_tom500_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val'],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the TOM500 dataset for multi-organ orbital segmentation in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'val'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_tom500_paths(path, split, download)

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


def get_tom500_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val'],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the TOM500 dataloader for multi-organ orbital segmentation in MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'val'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_tom500_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
