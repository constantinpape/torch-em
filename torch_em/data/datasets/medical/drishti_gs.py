"""The Drishti-GS dataset contains annotations for optic disc and optic cup
segmentation in Fundus images, for the task of glaucoma assessment.

The original data is hosted at https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php,
but downloading it (both the training and the test split) requires registering a team on that site.
This dataloader uses a mirror of the data hosted on Kaggle instead:
https://www.kaggle.com/datasets/lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation

The label masks are soft consensus maps (from four experts) stored as grayscale images, with values
scaled to `[0, 255]`. We binarize them here using a majority vote threshold (i.e. more than half of
the experts agree on the pixel belonging to the optic disc / cup).

The dataset is from the publication https://doi.org/10.1109/isbi.2014.6867807.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from pathlib import Path
from typing import Union, Tuple, Literal, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


KAGGLE_DATASET_NAME = "lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation"


def get_drishti_gs_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Drishti-GS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    if os.path.exists(os.path.join(path, "Training")) and os.path.exists(os.path.join(path, "Test")):
        return path

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name=KAGGLE_DATASET_NAME, download=download)
    zip_path = os.path.join(path, "drishtigs-retina-dataset-for-onh-segmentation.zip")
    util.unzip(zip_path=zip_path, dst=path)

    # The zip archive ships the training and test splits nested inside an extra, timestamped
    # directory level, e.g. 'Training-<timestamp>/Training'. Flatten this to 'Training'/'Test'.
    train_dir = glob(os.path.join(path, "Training*"))[0]
    os.rename(os.path.join(train_dir, "Training"), os.path.join(path, "Training"))
    os.rmdir(train_dir)

    test_dir = glob(os.path.join(path, "Test*"))[0]
    os.rename(os.path.join(test_dir, "Test"), os.path.join(path, "Test"))
    os.rmdir(test_dir)

    return path


def _binarize_mask(soft_map_path, gt_dir):
    dst_path = os.path.join(gt_dir, Path(soft_map_path).stem + ".tif")
    if os.path.exists(dst_path):
        return dst_path

    os.makedirs(gt_dir, exist_ok=True)
    soft_map = imageio.imread(soft_map_path)
    mask = (soft_map > 127).astype("uint8")
    imageio.imwrite(dst_path, mask)
    return dst_path


def get_drishti_gs_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    task: Literal["optic_disc", "optic_cup"] = "optic_disc",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Drishti-GS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        task: The choice of labels for the specific task.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_drishti_gs_data(path=path, download=download)

    assert split in ["train", "test"], f"'{split}' is not a valid split."
    assert task in ["optic_disc", "optic_cup"], f"'{task}' is not a valid task."

    split_dir = "Training" if split == "train" else "Test"
    gt_root = "GT" if split == "train" else "Test_GT"
    softmap_name = "ODsegSoftmap" if task == "optic_disc" else "cupsegSoftmap"

    image_paths = sorted(glob(os.path.join(data_dir, split_dir, "Images", "*", "*.png")))

    gt_dir = os.path.join(data_dir, split_dir, f"gt_{task}")
    gt_paths = []
    for image_path in image_paths:
        stem = Path(image_path).stem
        soft_map_path = os.path.join(data_dir, split_dir, gt_root, stem, "SoftMap", f"{stem}_{softmap_name}.png")
        gt_paths.append(_binarize_mask(soft_map_path, gt_dir))

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0

    return image_paths, gt_paths


def get_drishti_gs_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    task: Literal["optic_disc", "optic_cup"] = "optic_disc",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Drishti-GS dataset for segmentation of optic disc and optic cup in fundus images.

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
    image_paths, gt_paths = get_drishti_gs_paths(path, split, task, download)

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


def get_drishti_gs_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    task: Literal["optic_disc", "optic_cup"] = "optic_disc",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Drishti-GS dataloader for segmentation of optic disc and optic cup in fundus images.

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
    dataset = get_drishti_gs_dataset(path, patch_shape, split, task, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
