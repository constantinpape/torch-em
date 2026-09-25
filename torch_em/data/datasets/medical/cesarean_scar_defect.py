"""The Cesarean Scar Defect (CSD) dataset contains annotations for cesarean scar defect
segmentation in transvaginal ultrasound images.

The dataset consists of 501 images with binary segmentation masks (foreground: scar defect),
stored in a Pascal VOC-style layout ('JPEGImages' for the raw images, 'SegmentationClass' for
the label masks). The official 'ImageSets/Segmentation/train.txt' and 'val.txt' files list more
image ids than are actually shipped in the archive (802 and 507 respectively, out of 501 total
images); this module intersects the listed ids with the images that are actually present on disk
to build the 'train' (401 images) and 'val' (100 images) splits.

The data is located at https://doi.org/10.5281/zenodo.17789273, released under a CC-BY-4.0 license.

This dataset is from the publication https://arxiv.org/abs/2605.26774.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/17789273/files/VOCdevkit_CSD.zip"
CHECKSUM = "d3c37dd971260d7be2e50e07375f199a2da6e103141f4e4dc06789f25f983380"

SPLITS = ("train", "val")


def get_cesarean_scar_defect_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Cesarean Scar Defect dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the VOC-style data folder.
    """
    data_dir = os.path.join(path, "VOCdevkit_CSD", "VOCdevkit_CSD", "VOC2007")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "VOCdevkit_CSD.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    assert os.path.exists(data_dir), f"The extraction of the archive did not create the expected folder '{data_dir}'."

    return data_dir


def get_cesarean_scar_defect_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val"], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Cesarean Scar Defect data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'val'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose one of {SPLITS}.")

    data_dir = get_cesarean_scar_defect_data(path, download)

    with open(os.path.join(data_dir, "ImageSets", "Segmentation", f"{split}.txt")) as f:
        split_ids = {line.strip() for line in f if line.strip()}

    label_paths = natsorted(
        p for p in glob(os.path.join(data_dir, "SegmentationClass", "*.png"))
        if os.path.splitext(os.path.basename(p))[0] in split_ids
    )
    raw_paths = [
        os.path.join(data_dir, "JPEGImages", f"{os.path.splitext(os.path.basename(p))[0]}.jpg") for p in label_paths
    ]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in raw_paths)

    return raw_paths, label_paths


def get_cesarean_scar_defect_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Cesarean Scar Defect dataset for scar defect segmentation in transvaginal ultrasound.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'val'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cesarean_scar_defect_paths(path, split, download)

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
        **kwargs
    )


def get_cesarean_scar_defect_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Cesarean Scar Defect dataloader for scar defect segmentation in transvaginal ultrasound.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'val'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cesarean_scar_defect_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
