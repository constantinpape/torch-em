"""The PediMS dataset contains annotations for multiple sclerosis lesion segmentation in pediatric brain MRI.

The dataset comprises 28 longitudinal MRI exams from 9 pediatric MS patients (1 to 6 timepoints each),
acquired with T1-weighted, T2-weighted and T2-FLAIR sequences. Each timepoint ships a consensus lesion
mask, delineated and validated by senior clinical experts, in the native FLAIR space (the T1 and T2
scans are provided in their own native spaces and are not registered to the lesion mask, so this module
only exposes the FLAIR scan, which is used by this dataset for lesion delineation).
The label ids are: 0 = background, 1 = MS lesion.

The dataset is located at https://doi.org/10.6084/m9.figshare.28701065.v1 (CC BY 4.0).

This dataset is from the publication https://doi.org/10.1038/s41597-025-05346-5.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/articles/28701065/versions/1"
CHECKSUM = "2bd6dd209654a79247ba6340cf39afca7e6d20beacb61fea186432d22b05122a"

LABEL_IDS = {"background": 0, "ms_lesion": 1}


def get_pedims_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PediMS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "PediMS")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "pedims.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_pedims_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the PediMS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_pedims_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "P*", "T*", "processed", "brain_FLAIR.nii.gz")))
    label_paths = [p.replace("brain_FLAIR.nii.gz", "Consensus.nii") for p in raw_paths]

    assert len(raw_paths) == 28, f"Expected 28 timepoints, found {len(raw_paths)} in '{data_dir}'."
    for label_path in label_paths:
        assert os.path.exists(label_path), label_path

    return raw_paths, label_paths


def get_pedims_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PediMS dataset for pediatric multiple sclerosis lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_pedims_paths(path, download)

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


def get_pedims_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PediMS dataloader for pediatric multiple sclerosis lesion segmentation.

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
    dataset = get_pedims_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
