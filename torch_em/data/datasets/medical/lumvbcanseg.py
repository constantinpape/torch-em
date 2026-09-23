"""The LumVBCanSeg dataset contains annotations for lumbar vertebral body cancellous bone segmentation in CT.

The dataset consists of 185 lumbar CT scans acquired at ShengJing Hospital of China Medical University on
multiple Philips and Siemens scanners, resampled to an isotropic 1x1x1 mm resolution. Cases with vertebral
fractures, metal implants, bone tumors or foreign materials were excluded. Each scan has a corresponding
voxel-level segmentation mask covering the cancellous bone of the five lumbar vertebral bodies (L1-L5), with
labels 1-5 respectively. Annotations were made by 3 physicians and refined, dismissed or approved by a
physician with more than 30 years of experience in lumbar imaging.

The data is located at https://doi.org/10.5281/zenodo.8181250, released under a CC-BY-4.0 license.
NOTE: The archive is a single ~9.4 GB zip file with a flat 'Task908' folder, pairing each raw volume
('<case_id>.nii.gz') with its mask ('<case_id>_seg.nii.gz').

This dataset is from the publication https://doi.org/10.1016/j.compbiomed.2024.108237.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/8181250/files/LumVBCanSeg.zip/content"
CHECKSUM = "e41f1f5bae9611494997ac6f5f9d92e36ea80c4d6c417f98752932384f75f846"


def get_lumvbcanseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the LumVBCanSeg dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Task908")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "LumVBCanSeg.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_lumvbcanseg_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the LumVBCanSeg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_lumvbcanseg_data(path, download)

    label_paths = natsorted(glob(os.path.join(data_dir, "*_seg.nii.gz")))
    raw_paths = [p.replace("_seg.nii.gz", ".nii.gz") for p in label_paths]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in raw_paths)

    return raw_paths, label_paths


def get_lumvbcanseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LumVBCanSeg dataset for lumbar vertebral body cancellous bone segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_lumvbcanseg_paths(path, download)

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
        ndim=3,
        **kwargs
    )


def get_lumvbcanseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LumVBCanSeg dataloader for lumbar vertebral body cancellous bone segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lumvbcanseg_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
