"""The Tsakiroglou dataset contains annotations for nucleus segmentation in DAPI-channel
multiplex immunofluorescence images of follicular lymphoma tissue microarray cores.

Manual annotations have imperfect boundaries in places, and some patches show tiling
artifacts from stitching. Both are visible on inspection but do not prevent training use.

The dataset is located at https://doi.org/10.17632/nb46s9trx3.1.
This dataset is from the publication https://doi.org/10.1007/s00262-021-02945-0.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://data.mendeley.com/public-files/datasets/nb46s9trx3/files/e3252421-9a54-4db5-b835-1e55c184278b/file_downloaded"  # noqa
CHECKSUM = "1730dcaba538b03b8a1b1113242e64f59afebc9e2c4006ed559dc4251d4ade94"

SPLIT_FOLDERS = {"train": "training_validation", "test": "testing"}


def get_tsakiroglou_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Tsakiroglou dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the data is stored.
    """
    data_dir = os.path.join(path, "nuclear_segmentation_annotations 16bit")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "tsakiroglou.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_tsakiroglou_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Tsakiroglou data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLIT_FOLDERS:
        raise ValueError(f"'{split}' is not a valid split choice.")

    data_dir = get_tsakiroglou_data(path, download)
    split_dir = os.path.join(data_dir, SPLIT_FOLDERS[split])

    raw_paths = natsorted(glob(os.path.join(split_dir, "DAPI_images", "*.tif")))
    label_paths = natsorted(glob(os.path.join(split_dir, "labels 16bit", "*.tif")))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(
        os.path.basename(raw_path) == os.path.basename(label_path)
        for raw_path, label_path in zip(raw_paths, label_paths)
    )

    return raw_paths, label_paths


def get_tsakiroglou_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Tsakiroglou dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_tsakiroglou_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_tsakiroglou_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the Tsakiroglou dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_tsakiroglou_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
