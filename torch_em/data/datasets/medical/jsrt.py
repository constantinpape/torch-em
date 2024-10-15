"""The JSRT dataset contains annotations for lung segmentation
in chest X-Rays.

The database is located at http://db.jsrt.or.jp/eng.php
This dataset is from the publication https://doi.org/10.2214/ajr.174.1.1740071.
Please cite it if you use this dataset for a publication.
"""

import os
from glob import glob
from pathlib import Path
from typing import Optional, Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = {
    "Segmentation01": "http://imgcom.jsrt.or.jp/imgcom/wp-content/uploads/2018/11/Segmentation01.zip",
    "Segmentation02": "http://imgcom.jsrt.or.jp/imgcom/wp-content/uploads/2019/07/segmentation02.zip"
}

CHECKSUM = {
    "Segmentation01": "ab1f26a910bc18eae170928e9f2d98512cc4dc8949bf6cd38b98a93398714fcf",
    "Segmentation02": "f1432af4fcbd69342cf1bf2ca3d0d43b9535cdc6b160b86191b5b67de2fdbf3c"
}

ZIP_PATH = {
    "Segmentation01": "Segmentation01.zip",
    "Segmentation02": "segmentation02.zip"
}

DATA_DIR = {
    "Segmentation01": "Segmentation01",
    "Segmentation02": "segmentation02"
}


def get_jsrt_data(
    path: Union[os.PathLike, str], choice: Literal["Segmentation01", "Segmentation02"], download: bool = False
):
    """Download the JSRT dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        choice: The choice of data subset. Either 'Segmentation01' or 'Segmentation02'.
        download: Whether to download the data if it is not present.
    """
    data_dir = os.path.join(path, DATA_DIR[choice])
    if os.path.exists(data_dir):
        return

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, ZIP_PATH[choice])

    util.download_source(path=zip_path, url=URL[choice], download=download, checksum=CHECKSUM[choice])
    util.unzip(zip_path=zip_path, dst=path)


def get_jsrt_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'test'],
    choice: Optional[Literal['Segmentation01', 'Segmentation02']] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the JSRT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'train', or 'test'.
        choice: The choice of data subset. Either 'Segmentation01' or 'Segmentation02'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    available_splits = ["train", "test"]
    assert split in available_splits, f"{split} isn't a valid split choice. Please choose from {available_splits}."

    if choice is None:
        choice = list(URL.keys())
    else:
        if isinstance(choice, str):
            choice = [choice]

    image_paths, gt_paths = [], []
    for per_choice in choice:
        get_jsrt_data(path=path, download=download, choice=per_choice)

        if per_choice == "Segmentation01":
            root_dir = os.path.join(path, Path(ZIP_PATH[per_choice]).stem, split)
            all_image_paths = sorted(glob(os.path.join(root_dir, "org", "*.png")))
            all_gt_paths = sorted(glob(os.path.join(root_dir, "label", "*.png")))

        elif per_choice == "Segmentation02":
            root_dir = os.path.join(path, Path(ZIP_PATH[per_choice]).stem, "segmentation")
            all_image_paths = sorted(glob(os.path.join(root_dir, f"org_{split}", "*.bmp")))
            all_gt_paths = sorted(glob(os.path.join(root_dir, f"label_{split}", "*.png")))

        else:
            raise ValueError(f"{per_choice} is not a valid segmentation dataset choice.")

        image_paths.extend(all_image_paths)
        gt_paths.extend(all_gt_paths)

    assert len(image_paths) == len(gt_paths)

    return image_paths, gt_paths


def get_jsrt_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'],
    choice: Optional[Literal['Segmentation01', 'Segmentation02']] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the JSRT dataset for lung segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', or 'test'.
        choice: The choice of data subset. Either 'Segmentation01' or 'Segmentation02'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_jsrt_paths(path, split, choice, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths, raw_key=None, label_paths=gt_paths, label_key=None, patch_shape=patch_shape, **kwargs
    )


def get_jsrt_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: Literal['train', 'test'],
    choice: Optional[Literal['Segmentation01', 'Segmentation02']] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the JSRT dataloader for lung segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The data split to use. Either 'train', or 'test'.
        choice: The choice of data subset. Either 'Segmentation01' or 'Segmentation02'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_jsrt_dataset(path, patch_shape, split, choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
