"""The LyNSeC dataset contains annotations for nucleus segmentation
in IHC and H&E stained lymphoma tissue images.

The dataset is located at https://doi.org/10.5281/zenodo.8065174.
This dataset is from the publication https://doi.org/10.1016/j.compbiomed.2024.107978.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

import json
import numpy as np
import pandas as pd
import imageio.v3 as imageio
from sklearn.model_selection import train_test_split

import torch_em

from torch.utils.data import Dataset, DataLoader

from .. import util


URL = "https://zenodo.org/records/8065174/files/lynsec.zip"
CHECKSUM = "14b9b5a9c39cb41afc7f31de5a995cefff0947c215e14ab9c7a463f32fbbf4b6"


def _create_split_csv(path, data_dir, split, choice):
    assert split in ["train", "val", "test"], "Please choose a valid split."

    csv_path = os.path.join(path, f"lynsec_{choice}_split.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df[split] = df[split].apply(lambda x: json.loads(x.replace("'", '"')))  # ensures all items from column in list.
        split_list = df.iloc[0][split]

    else:
        print(f"Creating a new split file at '{csv_path}'.")
        image_names = [
            os.path.basename(image).split(".")[0] for image in glob(os.path.join(data_dir, choice, 'images', '*.tif'))
        ]

        # Create random splits per dataset.
        train_ids, test_ids = train_test_split(image_names, test_size=0.2)  # 20% for test split.
        train_ids, val_ids = train_test_split(train_ids, test_size=0.15)  # 15% for val split.
        split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

        df = pd.DataFrame.from_dict([split_ids])
        df.to_csv(csv_path, index=False)
        split_list = split_ids[split]

    return split_list


def _preprocess_dataset(data_dir):
    data_dirs = natsorted(glob(os.path.join(data_dir, "lynsec*")))
    for _dir in data_dirs:
        if os.path.basename(_dir) == "lynsec 1":
            target_dir = "ihc"
        else:
            target_dir = "h&e"

        image_dir = os.path.join(data_dir, target_dir, "images")
        label_dir = os.path.join(data_dir, target_dir, "labels")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        paths = natsorted(glob(os.path.join(_dir, "*.npy")))
        for fpath in tqdm(paths, desc="Preprocessing inputs"):
            fname = Path(fpath).stem
            darray = np.load(fpath)

            raw = darray[..., :3]
            labels = darray[..., 3]

            if target_dir == "h&e" and fname in [f"{i}_l2" for i in range(35)]:  # set of images have mismatching labels
                continue

            imageio.imwrite(os.path.join(image_dir, f"{fname}.tif"), raw, compression="zlib")
            imageio.imwrite(os.path.join(label_dir, f"{fname}.tif"), labels, compression="zlib")


def get_lynsec_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the LyNSeC dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the downloaded data.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(path, "lynsec.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir)

    _preprocess_dataset(data_dir)

    return data_dir


def get_lynsec_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "val", "test"]] = None,
    choice: Optional[Literal['ihc', 'h&e']] = None,
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the LyNSec data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        choice: The choice of dataset.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the image data.
        List of filepaths to the label data.
    """
    data_dir = get_lynsec_data(path, download)

    if choice is None:
        choice = "*"

    raw_paths = natsorted(glob(os.path.join(data_dir, choice, "images", "*.tif")))
    label_paths = natsorted(glob(os.path.join(data_dir, choice, "labels", "*.tif")))

    if split is not None:
        if choice == "*":  # If user did not choose a split, we make splits for both datasets.
            split_list = _create_split_csv(path, data_dir, split, "h&e")
            split_list.extend(_create_split_csv(path, data_dir, split, "ihc"))
        else:
            split_list = _create_split_csv(path, data_dir, split, choice)

        # Filter paths which are valid for the chosen split.
        raw_paths = [p for p in raw_paths if os.path.basename(p).split(".")[0] in split_list]
        label_paths = [p for p in label_paths if os.path.basename(p).split(".")[0] in split_list]

    return raw_paths, label_paths


def get_lynsec_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "val", "test"]] = None,
    choice: Optional[Literal['ihc', 'h&e']] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LyNSeC dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        choice: The choice of dataset.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_lynsec_paths(path, split, choice, download)

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
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_lynsec_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "val", "test"]] = None,
    choice: Optional[Literal['ihc', 'h&e']] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LyNSeC dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        choice: The choice of dataset.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lynsec_dataset(path, patch_shape, split, choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
