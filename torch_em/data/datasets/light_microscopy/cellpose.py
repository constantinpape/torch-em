"""This dataset contains annotation for cell segmentation in fluorescene microscently-labeled microscopy images.

This dataset is from the following publications:
- https://doi.org/10.1038/s41592-020-01018-x
- https://doi.org/10.1038/s41592-022-01663-4
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, Optional, List

import torch_em

from torch.utils.data import Dataset, DataLoader

from .. import util
from .neurips_cell_seg import to_rgb


AVAILABLE_CHOICES = ["cyto", "cyto2"]


def get_cellpose_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    choice: Literal["cyto", "cyto2"],
    download: bool = False,
) -> str:
    """Instruction to download CellPose data.

    NOTE: Please download the dataset from "https://www.cellpose.org/dataset".

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'train', or 'test'.
        choice: The choice of dataset. Either 'cyto' or 'cyto2'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder where the data is manually downloaded.
    """
    per_choice_dir = os.path.join(path, choice)  # path where the unzipped files will be stored
    if choice == "cyto":
        assert split in ["train", "test"], f"'{split}' is not a valid split in '{choice}'."
        zip_path = os.path.join(path, f"{split}.zip")
        data_dir = os.path.join(per_choice_dir, split)  # path where the per split images for 'cyto' exist.
    elif choice == "cyto2":
        assert split == "train", f"'{split}' is not a valid split in '{choice}'."
        zip_path = os.path.join(path, "train_cyto2.zip")
        data_dir = os.path.join(per_choice_dir, "train_cyto2")  # path where 'train' split images for 'cyto2' exist.
    else:
        raise ValueError(f"'{choice}' is not a valid dataset choice.")

    if os.path.exists(data_dir):
        return data_dir
    else:
        if not os.path.exists(zip_path) and download:
            raise NotImplementedError(
                "The dataset cannot be automatically downloaded. "
                "Please see 'get_cellpose_data' in 'torch_em/data/datasets/light_microscopy/cellpose.py' for details."
            )
        util.unzip(zip_path=zip_path, dst=per_choice_dir, remove=False)

    return data_dir


def get_cellpose_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'test'],
    choice: Optional[Literal["cyto", "cyto2"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CellPose data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', or 'test'.
        choice: The choice of dataset. Either 'cyto' or 'cyto2'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_cellpose_data(path, split, choice, download)

    image_paths = natsorted(glob(os.path.join(data_dir, "*_img.png")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "*_masks.png")))

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0

    return image_paths, gt_paths


def get_cellpose_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    choice: Optional[Literal["cyto", "cyto2"]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CellPose dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', or 'test'.
        choice: The choice of dataset. Either 'cyto' or 'cyto2'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert split in ["train", "test"]

    if choice is None:
        choice = AVAILABLE_CHOICES
    else:
        if not isinstance(choice, list):
            choice = [choice]

    image_paths, gt_paths = [], []
    for per_choice in choice:
        assert per_choice in AVAILABLE_CHOICES
        per_image_paths, per_gt_paths = get_cellpose_paths(path, split, per_choice, download)
        image_paths.extend(per_image_paths)
        gt_paths.extend(per_gt_paths)

    if "raw_transform" not in kwargs:
        kwargs["raw_transform"] = torch_em.transform.get_raw_transform(augmentation2=to_rgb)

    if "transform" not in kwargs:
        kwargs["transform"] = torch_em.transform.get_augmentations(ndim=2)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        ndim=2,
        **kwargs
    )


def get_cellpose_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    choice: Optional[Literal["cyto", "cyto2"]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CellPose dataloader for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', or 'test'.
        choice: The choice of dataset. Either 'cyto' or 'cyto2'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cellpose_dataset(path, patch_shape, split, choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
