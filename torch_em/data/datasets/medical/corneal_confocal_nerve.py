"""The Corneal Confocal Nerve dataset contains annotations for pixel-level corneal nerve segmentation
in in-vivo corneal confocal microscopy (CCM) images.

The dataset comprises 410 images from 88 participants, collected across two independently acquired
subsets with distinct acquisition conditions and participant groups. Each image is paired with an
expert-reviewed pixel-level nerve mask, in contrast to most other publicly available CCM datasets, which
only provide centreline annotations.

The dataset is located at https://doi.org/10.5281/zenodo.18779434 (CC BY 4.0).

This dataset is from the publication https://doi.org/10.1038/s41597-026-07418-6.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/18779434/files/Dataset.zip"
CHECKSUM = "34c657a250487db58c17b3de0407f1a38fc85923f6886f1a52c193c04c995595"


def get_corneal_confocal_nerve_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Corneal Confocal Nerve dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "Dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_corneal_confocal_nerve_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Corneal Confocal Nerve data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_corneal_confocal_nerve_data(path, download)

    image_paths = natsorted(glob(os.path.join(data_dir, "images", "*.png")))
    label_paths = [p.replace(os.sep + "images" + os.sep, os.sep + "annotations" + os.sep) for p in image_paths]

    assert len(image_paths) > 0, f"Could not find any images in '{data_dir}'."
    for label_path in label_paths:
        assert os.path.exists(label_path), label_path

    return image_paths, label_paths


def get_corneal_confocal_nerve_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Corneal Confocal Nerve dataset for nerve segmentation in CCM images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_corneal_confocal_nerve_paths(path, download)

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
        is_seg_dataset=False,
        **kwargs
    )


def get_corneal_confocal_nerve_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Corneal Confocal Nerve dataloader for nerve segmentation in CCM images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_corneal_confocal_nerve_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
