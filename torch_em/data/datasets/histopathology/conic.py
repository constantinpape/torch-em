
"""The CONIC dataset contains annotations for nucleus segmentation
in histopathology images in H&E stained colon tissue.

This dataset is from the publication https://doi.org/10.48550/arXiv.2303.06274.
Please cite it if you use this dataset for your research.
"""

import os
import numpy as np
from glob import glob
from typing import Tuple, Union, List, Literal
import gdown
from tqdm import tqdm

import pandas as pd

from torch.utils.data import Dataset, DataLoader

import torch_em

from torch_em.data.datasets import util


def _create_split_list(path, split):
    # this takes a split.csv with indices for train and test images
    # generated according to HoVerNet repo: https://github.com/vqdang/hover_net/blob/conic/generate_split.py 
    # We take the FOLD_IDX = 0 as used for the baseline model
    try:
        df = pd.read_csv(os.path.join(path, 'split.csv'))
    except FileNotFoundError:
        raise FileNotFoundError("split.csv with curator-defined split not found in dataset directory")

    split_list = [int(v) for v in df[split].dropna()]
    return split_list


def _extract_images(split, path):
    import h5py

    split_list = _create_split_list(path, split)

    images = np.load(os.path.join(path, "images.npy"))
    labels = np.load(os.path.join(path, "labels.npy"))

    instance_masks = []
    raw = []
    semantic_masks = []

    for idx, (image, label) in tqdm(enumerate(zip(images, labels)), desc=f"Extracting {split} data",
                                    total=images.shape[0]):
        if idx not in split_list:
            continue

        semantic_masks.append(label[:, :, 1])
        instance_masks.append(label[:, :, 0])
        raw.append(image)

    raw = np.stack(raw).transpose(3, 0, 1, 2)  # B, H, W, C --> C, B, H, W
    instance_masks = np.stack(instance_masks)
    semantic_masks = np.stack(semantic_masks)

    output_file = os.path.join(path, f"{split}.h5")
    with h5py.File(output_file, "a") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels/instance", data=instance_masks, compression="gzip")
        f.create_dataset("labels/semantic", data=semantic_masks, compression="gzip")


def get_conic_data(path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False):
    """Download the CONIC dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        download: Whether to download the data if it is not present.
    """
    if split not in ['train', 'test']:
        raise ValueError(f"'{split}' is not a valid split.")

    image_files = glob(os.path.join(path, "*.h5"))
    if len(image_files) > 0:
        return

    os.makedirs(path, exist_ok=True)

    # Load data if not in the given directory
    if not os.path.exists(os.path.join(path, "images.npy")) and download:
        gdown.download_folder("https://drive.google.com/drive/folders/1il9jG7uA4-ebQ_lNmXbbF2eOK9uNwheb", output=path,
                              quiet=False)
    # Extract and preprocess images for all splits
    for _split in ['train', 'test']:
        _extract_images(_split, path)


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
    get_conic_data(path, split, download)
    return os.path.join(path, f"{split}.h5")


def get_conic_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    label_choice: Literal["instance", "semantic"] = "instance",
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
    label_choice: Literal["instance", "semantic"] = "instance",
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
