"""The Mice TumSeg contains annotations for tumor segmentation in micro-CT scans.

This dataset is from the publication https://doi.org/10.1038/s41597-024-03814-y.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://erda.ku.dk/archives/ba4fcd9bfa0fb581d593297dd43d1fd1/TumSeg%20database.zip"
CHECKSUM = "1c0567358ec81b9e085434a0362c1c2981b791dc2733931aeb022174ecb80399"


def get_mice_tumseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Mice TumSeg dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, r"TumSeg database")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "TumSeg_database.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)
    return data_dir


def get_mice_tumseg_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'val', 'test'],
    rater: Literal["A", "B", "C", "STAPLE"] = "A",
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Mice TumSeg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        rater: The choice of annotator.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_mice_tumseg_data(path, download)

    if rater in ["A", "B", "C"]:
        ann_choice = f"Annotator_{rater}"
    elif rater == "STAPLE":
        ann_choice = rater
    else:
        raise ValueError(f"'{rater}' is not a valid rater choice.")

    raw_paths = natsorted(glob(os.path.join(data_dir, "Dataset*", "**", "CT*.nii.gz"), recursive=True))
    label_paths = natsorted(glob(os.path.join(data_dir, "Dataset*", "**", f"{ann_choice}*.nii.gz"), recursive=True))

    if split == "train":
        raw_paths, label_paths = raw_paths[:325], label_paths[:325]
    elif split == "val":
        raw_paths, label_paths = raw_paths[325:360], label_paths[325:360]
    elif split == "test":
        raw_paths, label_paths = raw_paths[360:], label_paths[360:]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_mice_tumseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    rater: Literal["A", "B", "C", "STAPLE"] = "A",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Mice TumSeg dataset for tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        rater: The choice of annotator.
        download: Whether to download the data if it is not present.
        resize_inputs: Whether to resize the inputs.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_mice_tumseg_paths(path, split, rater, download)

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
        **kwargs
    )


def get_mice_tumseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    rater: Literal["A", "B", "C", "STAPLE"] = "A",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Mice TumSeg dataloader for tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        rater: The choice of annotator.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mice_tumseg_dataset(path, patch_shape, split, rater, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
