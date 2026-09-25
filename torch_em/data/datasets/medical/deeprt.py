"""The DeepRT dataset contains annotations for retinal tissue segmentation in optical coherence
tomography (OCT) B-scans, originally released to support self-supervised pretraining for diabetic
retinopathy classification.

The dataset ships 1,286 OCT images (acquired with Spectralis and Topcon devices) with binary retinal
tissue masks, pre-split into 'train', 'validation' and 'test' (744 / 265 / 277 images respectively)
via the 'file_names_complete' mapping files shipped in the archive.
NOTE: the associated publication reports 1,009 "semantic segmentations", a smaller curated subset;
this loader exposes the full 1,286 labeled images shipped in the 'thickness_segmentation_data' archive.

The data is located at https://doi.org/10.5281/zenodo.3626020, released under a CC-BY-4.0 license.

This dataset is from the publication https://doi.org/10.1038/s42256-020-00247-1.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from typing import Union, Tuple, List, Literal

import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/3626020/files/thickness_segmentation_data.tar.gz"
CHECKSUM = "f0c3c0ce9f140e470f6579bc6ec634d719d5947726f8b38f48f04f88026cdaa8"

SPLITS = ["train", "validation", "test"]


def _binarize_labels(data_dir):
    label_paths = glob(os.path.join(data_dir, "data", "all_labels", "*.png"))
    for label_path in label_paths:
        label = np.array(Image.open(label_path).convert("L"))
        Image.fromarray((label > 0).astype("uint8")).save(label_path)


def get_deeprt_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the DeepRT dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "DeepRT")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    tar_path = os.path.join(path, "thickness_segmentation_data.tar.gz")
    util.download_source(path=tar_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip_tarfile(tar_path=tar_path, dst=data_dir, remove=False)

    _binarize_labels(data_dir)

    return data_dir


def get_deeprt_paths(
    path: Union[os.PathLike, str], split: Literal["train", "validation", "test"] = "train", download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the DeepRT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train', 'validation' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose one of {SPLITS}.")

    data_dir = get_deeprt_data(path, download)

    mapping = pd.read_csv(os.path.join(data_dir, "data", "file_names_complete", f"{split}_new_old_mapping.csv"))
    ids = mapping["new_id"].tolist()

    raw_paths = [os.path.join(data_dir, "data", "all_images", f"{i}.png") for i in ids]
    label_paths = [os.path.join(data_dir, "data", "all_labels", f"{i}.png") for i in ids]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in raw_paths) and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_deeprt_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "validation", "test"] = "train",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DeepRT dataset for retinal tissue segmentation in OCT images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'validation' or 'test'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_deeprt_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
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
        ndim=2,
        **kwargs
    )


def get_deeprt_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "validation", "test"] = "train",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DeepRT dataloader for retinal tissue segmentation in OCT images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'validation' or 'test'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_deeprt_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
