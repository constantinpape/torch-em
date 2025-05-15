"""The CONIC dataset contains annotations for nucleus segmentation
in histopathology images in H&E stained colon tissue.

This dataset is from the publication https://doi.org/10.1016/j.media.2023.103047.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from typing import Tuple, Union, List, Literal

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

import torch_em

from torch_em.data.datasets import util
from sklearn.model_selection import StratifiedShuffleSplit


URL = "https://drive.google.com/drive/folders/1il9jG7uA4-ebQ_lNmXbbF2eOK9uNwheb?usp=sharing"


def _create_split_list(path, split):
    # source: HoVerNet repo: https://github.com/vqdang/hover_net/blob/conic/generate_split.py.
    # We take the FOLD_IDX = 0 as used for the baseline model

    split_csv = os.path.join(path, "split.csv")

    if os.path.exists(split_csv):
        split_df = pd.read_csv(split_csv)
    else:
        SEED = 5
        info = pd.read_csv(os.path.join(path, "patch_info.csv"))
        file_names = np.squeeze(info.to_numpy()).tolist()

        img_sources = [v.split('-')[0] for v in file_names]
        img_sources = np.unique(img_sources)

        cohort_sources = [v.split('_')[0] for v in img_sources]
        _, cohort_sources = np.unique(cohort_sources, return_inverse=True)

        num_trials = 10
        splitter = StratifiedShuffleSplit(n_splits=num_trials, train_size=0.8, test_size=0.2, random_state=SEED)

        splits = {}
        split_generator = splitter.split(img_sources, cohort_sources)
        for train_indices, valid_indices in split_generator:
            train_cohorts = img_sources[train_indices]
            valid_cohorts = img_sources[valid_indices]

            assert np.intersect1d(train_cohorts, valid_cohorts).size == 0

            train_names = [
                file_name for file_name in file_names for source in train_cohorts if source == file_name.split('-')[0]
            ]
            valid_names = [
                file_name for file_name in file_names for source in valid_cohorts if source == file_name.split('-')[0]
            ]

            train_names = np.unique(train_names)
            valid_names = np.unique(valid_names)
            print(f'Train: {len(train_names):04d} - Valid: {len(valid_names):04d}')

            assert np.intersect1d(train_names, valid_names).size == 0

            train_indices = [file_names.index(v) for v in train_names]
            valid_indices = [file_names.index(v) for v in valid_names]

            while len(train_indices) > len(valid_indices):
                valid_indices.append(np.nan)

            splits['train'] = train_indices
            splits['test'] = valid_indices
            break

        split_df = pd.DataFrame(splits)
        split_df.to_csv(split_csv, index=False)

    split_list = [int(v) for v in split_df[split].dropna()]
    return split_list


def _extract_images(split, path):

    split_list = _create_split_list(path, split)

    images = np.load(os.path.join(path, "images.npy"))
    labels = np.load(os.path.join(path, "labels.npy"))

    instance_masks = []
    raw = []
    semantic_masks = []

    for idx, (image, label) in tqdm(
        enumerate(zip(images, labels)), desc=f"Extracting '{split}' data", total=images.shape[0]
    ):
        if idx not in split_list:
            continue

        semantic_masks.append(label[:, :, 1])
        instance_masks.append(label[:, :, 0])
        raw.append(image)

    raw = np.stack(raw).transpose(3, 0, 1, 2)  # B, H, W, C --> C, B, H, W
    instance_masks = np.stack(instance_masks)
    semantic_masks = np.stack(semantic_masks)

    import h5py
    with h5py.File(os.path.join(path, f"{split}.h5"), "a") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels/instances", data=instance_masks, compression="gzip")
        f.create_dataset("labels/semantic", data=semantic_masks, compression="gzip")


def get_conic_data(path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False) -> str:
    """Download the CONIC dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is download for further processing.
    """
    if split not in ['train', 'test']:
        raise ValueError(f"'{split}' is not a valid split.")

    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir) and glob(os.path.join(data_dir, "*.h5")):
        return data_dir

    os.makedirs(path, exist_ok=True)

    # Download the files from google drive.
    util.download_source_gdrive(path=data_dir, url=URL, download=download, download_type="folder", quiet=False)

    # Extract and preprocess images for all splits
    for _split in ['train', 'test']:
        _extract_images(_split, data_dir)

    return data_dir


def get_conic_paths(
    path: Union[os.PathLike], split: Literal["train", "test"], download: bool = False
) -> List[str]:
    """Get paths to the CONIC data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data splits.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    data_dir = get_conic_data(path, split, download)
    return os.path.join(data_dir, f"{split}.h5")


def get_conic_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    label_choice: Literal["instances", "semantic"] = "instances",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CONIC dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the input images.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    data_paths = get_conic_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="raw",
        label_paths=data_paths,
        label_key=f"labels/{label_choice}",
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs
    )


def get_conic_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    label_choice: Literal["instances", "semantic"] = "instances",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CONIC dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_conic_dataset(path, patch_shape, split, label_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(ds, batch_size, **loader_kwargs)
