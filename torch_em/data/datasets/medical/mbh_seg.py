"""The MBH Seg dataset contains annotations for intracranial hemorrhages
in non-contrast CT scans.

This dataset is from the MBH-Seg challenge: https://mbh-seg.com
- original scans: https://kaggle.com/competitions/rsna-intracranial-hemorrhage-detection
Please cite these if you use this dataset for your publication.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://huggingface.co/datasets/WuBiao/BHSD/resolve/main/label_192.zip"
CHECKSUM = "582bf184af993541a4958a4d209a6a44e3bbe702a5daefaf9fb1733a4e7a6e39"


def get_mbh_seg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the MBH Seg dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is download.
    """
    data_dir = os.path.join(path, "label_192")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "label_192.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_mbh_seg_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the MBH Seg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_mbh_seg_data(path=path, download=download)
    image_paths = natsorted(glob(os.path.join(data_dir, "images", "*.nii.gz")))
    gt_paths = natsorted(glob(os.path.join(data_dir, r"ground truths", "*.nii.gz")))
    return image_paths, gt_paths


def get_mbh_seg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MBH Seg dataset for intracranial hemorrhage segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_mbh_seg_paths(path=path, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        **kwargs
    )


def get_mbh_seg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MBH Seg dataloader for intracranial hemorrhage segmentation.

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
    dataset = get_mbh_seg_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
