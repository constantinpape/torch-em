"""The CryoNuSeg dataset contains annotations for nucleus segmentation
in cryosectioned H&E stained histological images of 10 different organs.

This dataset is from the publication https://doi.org/10.1016/j.compbiomed.2021.104349.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import json
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def _create_split_csv(path, data_dir, split):
    csv_path = os.path.join(path, 'cryonuseg_split.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df[split] = df[split].apply(lambda x: json.loads(x.replace("'", '"')))  # ensures all items from column in list.
        split_list = df.iloc[0][split]

    else:
        print(f"Creating a new split file at '{csv_path}'.")
        image_names = [
            os.path.basename(image).split(".")[0] for image in glob(os.path.join(path, data_dir, '*.tif'))
        ]

        # Create random splits per dataset.
        train_ids, test_ids = train_test_split(image_names, test_size=0.2)  # 20% for test split.
        train_ids, val_ids = train_test_split(train_ids, test_size=0.15)  # 15% for val split.
        split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

        df = pd.DataFrame.from_dict([split_ids])
        df.to_csv(csv_path, index=False)

        split_list = split_ids[split]

    return split_list


def get_cryonuseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CryoNuSeg dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The folder where the data is downloaded and preprocessed.
    """
    data_dir = os.path.join(path, r"tissue images")
    if os.path.exists(os.path.join(path, r"tissue images")):
        return data_dir

    os.makedirs(path, exist_ok=True)
    util.download_source_kaggle(
        path=path, dataset_name="ipateam/segmentation-of-nuclei-in-cryosectioned-he-images", download=download
    )

    zip_path = os.path.join(path, "segmentation-of-nuclei-in-cryosectioned-he-images.zip")
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_cryonuseg_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    rater_choice: Literal["b1", "b2", "b3"] = "b1",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CryoNuSeg data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        rater: The choice of annotator.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the image data.
        List of filepaths to the label data.
    """
    data_dir = get_cryonuseg_data(path, download)

    if rater_choice == "b1":
        label_dir = r"Annotator 1 (biologist)/"
    elif rater_choice == "b2":
        label_dir = r"Annotator 1 (biologist second round of manual marks up)/" * 2
    elif rater_choice == "b3":
        label_dir = r"Annotator 2 (bioinformatician)/" * 2
    else:
        raise ValueError(f"'{rater_choice}' is not a valid rater choice.")

    # Point to the instance labels folder
    label_dir += r"label masks modify"
    split_list = _create_split_csv(path, label_dir, split)

    # Get the raw and label paths
    label_paths = natsorted([os.path.join(path, label_dir, f'{fname}.tif') for fname in split_list])
    raw_paths = natsorted([os.path.join(data_dir, f'{fname}.tif') for fname in split_list])

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_cryonuseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    rater: Literal["b1", "b2", "b3"] = "b1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CryoNuSeg dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        rater: The choice of annotator.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cryonuseg_paths(path, split, rater, download)

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


def get_cryonuseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    rater: Literal["b1", "b2", "b3"] = "b1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CryoNuSeg dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        rater: The choice of annotator.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cryonuseg_dataset(path, patch_shape, split, rater, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
