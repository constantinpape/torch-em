"""The TrackRAD dataset contains annotations for tumor segmentation in sagittal cine-MRI sequences,
acquired during MRI-guided radiotherapy treatments.

This is the training data of the TrackRAD2025 challenge for real-time tumor tracking. It provides
sagittal 2D cine-MRI sequences (a time-resolved stack of 2D frames per patient) from 585 patients,
acquired at six international centers on 0.35T (ViewRay MRIdian) or 1.5T (Elekta Unity) MRI-linacs,
with tumors in the thorax, abdomen and pelvis. For each labeled patient, every frame of the cine-MRI
sequence has a corresponding per-pixel tumor segmentation mask (`_labels.mha`), and the first frame
also has a separate single-frame mask (`_first_label.mha`).

NOTE: The challenge also ships a much larger 'unlabeled' collection (over 2.8 million frames from 477
patients) that has no segmentation masks. It is not supported here, since it cannot be used for
segmentation training.

NOTE: The raw and label volumes are stored as (height, width, num_frames), i.e. the time axis is the
last axis, not the first one; take this into account when choosing `patch_shape`.

The dataset is located at https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/TrackRAD2025
(DOI: 10.57967/hf/4539) and is distributed under the CC BY-NC 4.0 license.
This dataset is from the publication https://doi.org/10.1002/mp.17964.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


REPO_ID = "LMUK-RADONC-PHYS-RES/TrackRAD2025"

SPLITS = {
    "training": "trackrad2025_labeled_training_data",
    "pre-testing": "trackrad2025_labeled_pre-testing_data",
    "testing": "trackrad2025_labeled_testing_data",
}
"""Mapping from the split choice to its folder in the release."""


def get_trackrad_data(
    path: Union[os.PathLike, str], split: Literal["training", "pre-testing", "testing"] = "training",
    download: bool = False,
) -> str:
    """Download the TrackRAD dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLITS.keys())}.")

    data_dir = os.path.join(path, SPLITS[split])
    if os.path.exists(data_dir):
        return data_dir

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    from huggingface_hub import snapshot_download

    os.makedirs(path, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", local_dir=path, allow_patterns=f"{SPLITS[split]}/*"
    )
    return data_dir


def get_trackrad_paths(
    path: Union[os.PathLike, str], split: Literal["training", "pre-testing", "testing"] = "training",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the TrackRAD data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_trackrad_data(path=path, split=split, download=download)

    image_paths = natsorted(glob(os.path.join(data_dir, "*", "images", "*_frames.mha")))
    label_paths = natsorted(glob(os.path.join(data_dir, "*", "targets", "*_labels.mha")))

    assert len(image_paths) == len(label_paths) and len(image_paths) > 0

    return image_paths, label_paths


def get_trackrad_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["training", "pre-testing", "testing"] = "training",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the TrackRAD dataset for tumor segmentation in cine-MRI sequences.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_trackrad_paths(path=path, split=split, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs
    )


def get_trackrad_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["training", "pre-testing", "testing"] = "training",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the TrackRAD dataloader for tumor segmentation in cine-MRI sequences.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_trackrad_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
