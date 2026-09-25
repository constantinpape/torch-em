"""The Full-Head MRI Segmentation dataset contains annotations for whole-head segmentation in T1-weighted MRI,
including clinical cases with abnormal brain anatomy.

The dataset consists of 68 anonymized clinical subjects (with aphasia or apraxia after a stroke, plus a few
healthy controls, scanned at three institutions) and 4 additional healthy control subjects, each with a manually
corrected segmentation of the following 7 tissue classes: background, skin/scalp, skull, CSF, gray matter,
white matter and air (air cavities and extracephalic air, not always separated into two classes).

The dataset is located at
https://www.kaggle.com/datasets/andrewbirnbaum/full-head-mri-and-segmentation-of-stroke-patients
and is distributed under the CC BY-NC-SA 4.0 license.

This dataset is from the publication https://doi.org/10.1117/1.JMI.12.5.054001. Please cite it if you use this
dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


KAGGLE_DATASET = "andrewbirnbaum/full-head-mri-and-segmentation-of-stroke-patients"

LABEL_IDS = {
    "background": 0, "skin_scalp": 1, "skull": 2, "csf": 3, "gray_matter": 4, "white_matter": 5, "air": 6,
}


def get_full_head_mri_segmentation_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Full-Head MRI Segmentation dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Data")
    if os.path.exists(data_dir):
        return path

    os.makedirs(path, exist_ok=True)
    util.download_source_kaggle(path=path, dataset_name=KAGGLE_DATASET, download=download)

    zip_paths = glob(os.path.join(path, "*.zip"))
    assert len(zip_paths) > 0, f"Could not find the downloaded zip file at '{path}'."
    util.unzip(zip_path=zip_paths[0], dst=path)

    return path


def get_full_head_mri_segmentation_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Full-Head MRI Segmentation data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_full_head_mri_segmentation_data(path, download)

    raw_paths, label_paths = [], []

    # The anonymized clinical subjects (T1-weighted MRI file names end in '_deface.nii').
    for raw_path in natsorted(glob(os.path.join(
        data_dir, "Data", "Anonymized_Subjects", "T1-Weighted MRI", "*_deface.nii"
    ))):
        label_path = os.path.join(
            data_dir, "Data", "Anonymized_Subjects", "Full-Head Segmentation",
            os.path.basename(raw_path).replace("_deface.nii", "_label_deface.nii"),
        )
        if os.path.exists(label_path):
            raw_paths.append(raw_path)
            label_paths.append(label_path)

    # The healthy control subjects.
    for raw_path in natsorted(glob(os.path.join(data_dir, "Data", "Control_Subjects", "T1-Weighted MRI", "*.nii"))):
        label_path = os.path.join(
            data_dir, "Data", "Control_Subjects", "Full-Head Segmentation",
            os.path.basename(raw_path).replace(".nii", "_label.nii"),
        )
        if os.path.exists(label_path):
            raw_paths.append(raw_path)
            label_paths.append(label_path)

    assert len(raw_paths) > 0 and len(raw_paths) == len(label_paths)
    return raw_paths, label_paths


def get_full_head_mri_segmentation_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Full-Head MRI Segmentation dataset for whole-head segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_full_head_mri_segmentation_paths(path, download)

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


def get_full_head_mri_segmentation_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Full-Head MRI Segmentation dataloader for whole-head segmentation.

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
    dataset = get_full_head_mri_segmentation_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
