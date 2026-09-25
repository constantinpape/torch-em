"""The FIVES dataset contains annotations for retinal vessel segmentation in high-resolution
fundus images across four categories: normal, age-related macular degeneration (AMD), diabetic
retinopathy (DR) and glaucoma.

This dataset is from the publication https://doi.org/10.1038/s41597-022-01564-3.
The dataset is hosted on figshare at https://doi.org/10.6084/m9.figshare.19688169.v1 and is
licensed under CC BY 4.0. Please cite the publication above if you use this dataset for your
research.
"""

import os
from glob import glob
from pathlib import Path
from typing import Union, Tuple, Literal, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/34969398"
CHECKSUM = "be72f9af286b107bcebcc08a9dae7fc55c3fb0959409b689e14c72f9fdc4ad8e"

CATEGORIES = {"N": "normal", "A": "amd", "D": "dr", "G": "glaucoma"}


def get_fives_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the FIVES dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "FIVES A Fundus Image Dataset for AI-based Vessel Segmentation")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    rar_path = os.path.join(path, "fives.rar")
    util.download_source(path=rar_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip_rarfile(rar_path=rar_path, dst=path)

    return data_dir


def _get_fives_ground_truth(data_dir, split):
    gt_paths = sorted(glob(os.path.join(data_dir, split, "Ground truth", "*.png")))

    neu_gt_dir = os.path.join(data_dir, split, "gt")
    if os.path.exists(neu_gt_dir):
        return sorted(glob(os.path.join(neu_gt_dir, "*.tif")))
    else:
        os.makedirs(neu_gt_dir, exist_ok=True)

    neu_gt_paths = []
    for gt_path in gt_paths:
        gt = imageio.imread(gt_path)
        if gt.ndim == 3:
            gt = gt[..., 0]
        neu_gt_path = os.path.join(neu_gt_dir, Path(os.path.split(gt_path)[-1]).with_suffix(".tif"))
        imageio.imwrite(neu_gt_path, (gt > 0).astype("uint8"))
        neu_gt_paths.append(neu_gt_path)

    return sorted(neu_gt_paths)


def get_fives_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    category: Literal["normal", "amd", "dr", "glaucoma", "all"] = "all",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the FIVES data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'test'.
        category: The choice of disease category. One of 'normal', 'amd', 'dr', 'glaucoma' or 'all'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in ("train", "test"):
        raise ValueError(f"'{split}' is not a valid split.")

    if category == "all":
        prefixes = list(CATEGORIES)
    else:
        matches = [k for k, v in CATEGORIES.items() if v == category]
        if not matches:
            valid = list(CATEGORIES.values()) + ["all"]
            raise ValueError(f"'{category}' is not a valid category. Choose from {valid}.")
        prefixes = matches

    data_dir = get_fives_data(path=path, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, split, "Original", "*.png")))
    gt_paths = _get_fives_ground_truth(data_dir, split)

    image_paths = [p for p in image_paths if os.path.splitext(os.path.basename(p))[0].split("_")[-1] in prefixes]
    gt_paths = [p for p in gt_paths if os.path.splitext(os.path.basename(p))[0].split("_")[-1] in prefixes]

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0

    return image_paths, gt_paths


def get_fives_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    category: Literal["normal", "amd", "dr", "glaucoma", "all"] = "all",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the FIVES dataset for segmentation of retinal blood vessels in high-resolution fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        category: The choice of disease category.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_fives_paths(path=path, split=split, category=category, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_fives_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    category: Literal["normal", "amd", "dr", "glaucoma", "all"] = "all",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the FIVES dataloader for segmentation of retinal blood vessels in high-resolution fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        category: The choice of disease category.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_fives_dataset(path, patch_shape, split, category, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
