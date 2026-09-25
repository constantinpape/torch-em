"""The REFUGE dataset contains annotations for optic disc and optic cup segmentation
in Fundus images, for the task of glaucoma assessment.

The dataset was published in the "Retinal Fundus Glaucoma Challenge" (REFUGE), organized as part
of the 5th MICCAI Workshop on Ophthalmic Medical Image Analysis (OMIA) at MICCAI 2018.
It comprises 1200 fundus images (400 for training, validation and test each), with manual pixel-wise
annotations of the optic disc and optic cup, obtained by merging the annotations of seven independent
glaucoma specialists from the Zhongshan Ophthalmic Center, Sun Yat-sen University, China.

The original data is hosted at https://refuge.grand-challenge.org, but this hosting has become stale.
This dataloader uses a mirror of the data hosted on Kaggle: https://www.kaggle.com/datasets/victorlemosml/refuge2
(the 'REFUGE2' folder in this mirror corresponds to the original 2018 REFUGE data, not the REFUGE2 challenge).

The label masks are grayscale images (bmp for the train and test splits, png for the validation split) with
3 pixel values: 0 (optic cup), 128 (optic disc, excluding the cup) and 255 (background).

The dataset is from the publication https://doi.org/10.1016/j.media.2019.101570.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Union, Tuple, Literal, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


DATASET_NAME = "victorlemosml/refuge2"


def get_refuge_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the REFUGE dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "REFUGE2")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name=DATASET_NAME, download=download)
    zip_path = os.path.join(path, "refuge2.zip")
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _preprocess_labels(data_dir, mask_paths, task):
    gt_dir = os.path.join(data_dir, f"gt_{task}")
    os.makedirs(gt_dir, exist_ok=True)

    gt_paths = []
    for mask_path in tqdm(mask_paths, desc=f"Preprocessing labels for '{task}'"):
        gt_path = os.path.join(gt_dir, f"{Path(mask_path).stem}.tif")
        gt_paths.append(gt_path)
        if os.path.exists(gt_path):
            continue

        mask = imageio.imread(mask_path)
        if task == "disc":  # The optic disc region includes the optic cup.
            labels = (mask < 255).astype("uint8")
        else:  # The optic cup is the innermost region, marked with the pixel value 0.
            labels = (mask == 0).astype("uint8")

        imageio.imwrite(gt_path, labels)

    return gt_paths


def get_refuge_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    task: Literal["disc", "cup"] = "disc",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the REFUGE data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        task: The choice of labels for the specific task.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_refuge_data(path=path, download=download)

    assert split in ["train", "val", "test"], f"'{split}' is not a valid split."
    assert task in ["disc", "cup"], f"'{task}' is not a valid task."

    image_paths = sorted(glob(os.path.join(data_dir, split, "images", "*.jpg")))
    mask_paths = sorted(
        glob(os.path.join(data_dir, split, "mask", "*.bmp")) + glob(os.path.join(data_dir, split, "mask", "*.png"))
    )
    assert len(image_paths) == len(mask_paths) and len(image_paths) > 0

    gt_paths = _preprocess_labels(data_dir, mask_paths, task)

    return image_paths, gt_paths


def get_refuge_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    task: Literal["disc", "cup"] = "disc",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the REFUGE dataset for segmentation of optic disc and optic cup in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        task: The choice of labels for the specific task.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_refuge_paths(path, split, task, download)

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


def get_refuge_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    task: Literal["disc", "cup"] = "disc",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the REFUGE dataloader for segmentation of optic disc and optic cup in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        task: The choice of labels for the specific task.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_refuge_dataset(path, patch_shape, split, task, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
