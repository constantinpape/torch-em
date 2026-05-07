"""The ABUS dataset contains annotations for breast cancer segmentation in ultrasound images.

This dataset is located at https://www.kaggle.com/datasets/mohammedtgadallah/mt-small-dataset.
The dataset is from the publication https://doi.org/10.1371/journal.pone.0251899.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Tuple, List, Union, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def get_abus_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ABUS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "MT_Small_Dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name="mohammedtgadallah/mt-small-dataset", download=download)
    util.unzip(zip_path=os.path.join(path, "mt-small-dataset.zip"), dst=path)

    return data_dir


def get_abus_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    category: Literal["benign", "malign"],
    image_choice: Literal["raw", "fuzzy"] = "raw",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the ABUS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        category: The choice of tumor category.
        image_choice: The choice of input data.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_abus_data(path, download)

    if image_choice not in ["raw", "fuzzy"]:
        raise ValueError("Invalid input choice provided.", image_choice)

    if split not in ["train", "val", "test"]:
        raise ValueError("Invalid split choice provided.")

    if category not in ["benign", "malign"]:
        raise ValueError("Invalid tumor category provided.")

    cname = "Benign" if category == "benign" else "Malignant"
    raw_iname = f"Original_{cname}" if image_choice == "raw" else f"Fuzzy_{cname}"
    gt_iname = f"Ground_Truth_{cname}"

    image_paths = natsorted(glob(os.path.join(data_dir, cname, raw_iname, "*.png")))
    gt_paths = natsorted(glob(os.path.join(data_dir, cname, gt_iname, "*.png")))

    assert len(image_paths) and len(image_paths) == len(gt_paths)

    if split == "train":
        image_paths, gt_paths = image_paths[:125], gt_paths[:125]
    elif split == "val":
        image_paths, gt_paths = image_paths[125:150], gt_paths[125:150]
    else:
        image_paths, gt_paths = image_paths[150:], gt_paths[150:]

    return image_paths, gt_paths


def get_abus_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    category: Literal["benign", "malign"],
    split: Literal["train", "val", "test"],
    image_choice: Literal["raw", "fuzzy"] = "raw",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ABUS dataset for breast cancer segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        category: The choice of tumor category.
        split: The choice of data split.
        image_choice: The choice of input data.
        resize_inputs: Whether to resize the inputs.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_abus_paths(path, split, category, image_choice, download)

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
        ndim=2,
        with_channels=True,
        is_seg_dataset=False,
        **kwargs
    )


def get_abus_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    category: Literal["benign", "malign"],
    split: Literal["train", "val", "test"],
    image_choice: Literal["raw", "fuzzy"] = "raw",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ABUS dataloader for breast cancer segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        category: The choice of tumor category.
        split: The choice of data split.
        image_choice: The choice of input data.
        resize_inputs: Whether to resize the inputs.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_abus_dataset(path, patch_shape, category, split, image_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
