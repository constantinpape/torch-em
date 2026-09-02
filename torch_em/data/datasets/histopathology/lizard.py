"""The Lizard dataset contains annotations for nucleus segmentation
in histopathology images in H&E stained colon tissue.

This dataset is from the publication https://doi.org/10.48550/arXiv.2108.11195.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from shutil import rmtree
from natsort import natsorted
from typing import Tuple, Union, List, Literal

import pandas as pd
import imageio.v3 as imageio
from scipy.io import loadmat

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


SPLIT_MAP = {"train": 1, "val": 2, "test": 3}


def _create_split_list(path, split):
    df = pd.read_csv(os.path.join(path, 'lizard_labels', 'Lizard_Labels', 'info.csv'))
    split_list = [df['Filename'].iloc[i] for i in df.index if df['Split'].iloc[i] == SPLIT_MAP[split]]
    return split_list


def _extract_images(split, image_folder, label_folder, output_dir):
    import h5py

    image_files = glob(os.path.join(image_folder, "*.png"))
    split_list = _create_split_list(output_dir, split)
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    for image_file in tqdm(image_files, desc=f"Extract images from {os.path.abspath(image_folder)}"):
        fname = Path(os.path.basename(image_file))
        if fname.stem not in split_list:
            continue

        label_file = os.path.join(label_folder, fname.with_suffix(".mat"))
        assert os.path.exists(label_file), label_file

        image = imageio.imread(image_file)
        assert image.ndim == 3 and image.shape[-1] == 3

        labels = loadmat(label_file)
        segmentation = labels["inst_map"]
        assert image.shape[:-1] == segmentation.shape
        classes = labels["class"]

        image = image.transpose((2, 0, 1))
        assert image.shape[1:] == segmentation.shape

        output_file = os.path.join(output_dir, split, fname.with_suffix(".h5"))
        with h5py.File(output_file, "a") as f:
            f.create_dataset("image", data=image, compression="gzip")
            f.create_dataset("labels/segmentation", data=segmentation, compression="gzip")
            f.create_dataset("labels/classes", data=classes, compression="gzip")


def get_lizard_data(path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False):
    """Download the Lizard dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        download: Whether to download the data if it is not present.
    """
    if split not in SPLIT_MAP.keys():
        raise ValueError(f"'{split}' is not a valid split.")

    image_files = glob(os.path.join(path, split, "*.h5"))
    if len(image_files) > 0:
        return

    os.makedirs(path, exist_ok=True)
    util.download_source_kaggle(path=path, dataset_name="aadimator/lizard-dataset", download=download)
    zip_path = os.path.join(path, "lizard-dataset.zip")
    util.unzip(zip_path=zip_path, dst=path)

    image_folder1 = os.path.join(path, "lizard_images1", "Lizard_Images1")
    image_folder2 = os.path.join(path, "lizard_images2",  "Lizard_Images2")
    label_folder = os.path.join(path, "lizard_labels", "Lizard_Labels")

    assert os.path.exists(image_folder1), image_folder1
    assert os.path.exists(image_folder2), image_folder2
    assert os.path.exists(label_folder), label_folder

    # Extract and preprocess images for all splits
    for _split in SPLIT_MAP.keys():
        _extract_images(_split, image_folder1, os.path.join(label_folder, "Labels"), path)
        _extract_images(_split, image_folder2, os.path.join(label_folder, "Labels"), path)

    rmtree(os.path.join(path, "lizard_images1"))
    rmtree(os.path.join(path, "lizard_images2"))
    rmtree(os.path.join(path, "lizard_labels"))
    rmtree(os.path.join(path, "overlay"))


def get_lizard_paths(
    path: Union[os.PathLike], split: Literal["train", "val", "test"], download: bool = False
) -> List[str]:
    """Get paths to the Lizard data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data splits.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    get_lizard_data(path, split, download)
    data_paths = natsorted(glob(os.path.join(path, split, "*.h5")))
    return data_paths


def get_lizard_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Lizard dataset for nucleus segmentation.

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
    data_paths = get_lizard_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="image",
        label_paths=data_paths,
        label_key="labels/segmentation",
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs
    )


# TODO implement loading the classification labels
# TODO implement selecting different tissue types
def get_lizard_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Lizard dataloader for nucleus segmentation.

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
    ds = get_lizard_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(ds, batch_size, **loader_kwargs)
