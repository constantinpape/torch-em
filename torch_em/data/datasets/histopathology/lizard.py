"""The Lizard dataset contains annotations for nucleus segmentation
in histopathology images in H&E stained colon tissue.

This dataset is from the publication https://doi.org/10.48550/arXiv.2108.11195.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from shutil import rmtree
from typing import Tuple, Union, List
import pandas as pd
import imageio.v3 as imageio

from scipy.io import loadmat

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def create_split_list(path, split):
    df = pd.read_csv(os.path.join(path, 'lizard_labels', 'Lizard_Labels/info.csv'))
    split_list = []
    for i in df.index:
        image_split = df['Split'].iloc[i]
        if image_split == int(split[-1]):
            split_list.append(df['Filename'].iloc[i])
    return split_list


def _extract_images(image_folder, label_folder, output_dir, split):
    import h5py

    image_files = glob(os.path.join(image_folder, "*.png"))
    split_list = create_split_list(output_dir, split)
    output_path = os.path.join(output_dir, split)
    os.makedirs(output_path, exist_ok=True)
    for image_file in tqdm(image_files, desc=f"Extract images from {image_folder}"):
        fname = os.path.basename(image_file)
        if os.path.splitext(fname)[0] not in split_list:
            continue
        label_file = os.path.join(label_folder, fname.replace(".png", ".mat"))
        assert os.path.exists(label_file), label_file

        image = imageio.imread(image_file)
        assert image.ndim == 3 and image.shape[-1] == 3

        labels = loadmat(label_file)
        segmentation = labels["inst_map"]
        assert image.shape[:-1] == segmentation.shape
        classes = labels["class"]

        image = image.transpose((2, 0, 1))
        assert image.shape[1:] == segmentation.shape

        output_file = os.path.join(output_path, fname.replace(".png", ".h5"))
        with h5py.File(output_file, "a") as f:
            f.create_dataset("image", data=image, compression="gzip")
            f.create_dataset("labels/segmentation", data=segmentation, compression="gzip")
            f.create_dataset("labels/classes", data=classes, compression="gzip")


def get_lizard_data(path, download, split):
    """Download the Lizard dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.
    """
    util.download_source_kaggle(path=path, dataset_name="aadimator/lizard-dataset", download=download)
    zip_path = os.path.join(path, "lizard-dataset.zip")
    util.unzip(zip_path=zip_path, dst=path)

    image_files = glob(os.path.join(path, split, "*.h5"))
    if len(image_files) > 0:
        return

    os.makedirs(path, exist_ok=True)

    image_folder1 = os.path.join(path, "lizard_images1", "Lizard_Images1")
    image_folder2 = os.path.join(path, "lizard_images2",  "Lizard_Images2")
    label_folder = os.path.join(path, "lizard_labels", "Lizard_Labels")

    assert os.path.exists(image_folder1), image_folder1
    assert os.path.exists(image_folder2), image_folder2
    assert os.path.exists(label_folder), label_folder

    _extract_images(image_folder1, os.path.join(label_folder, "Labels"), path, split)
    _extract_images(image_folder2, os.path.join(label_folder, "Labels"), path, split)

    rmtree(os.path.join(path, "lizard_images1"))
    rmtree(os.path.join(path, "lizard_images2"))
    rmtree(os.path.join(path, "lizard_labels"))
    rmtree(os.path.join(path, "overlay"))


def get_lizard_paths(path: Union[os.PathLike], split, download: bool = False) -> List[str]:
    """Get paths to the Lizard data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    get_lizard_data(path, download, split)

    data_paths = glob(os.path.join(path, split, "*.h5"))
    data_paths.sort()
    return data_paths


def get_lizard_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, int], split, download: bool = False, **kwargs
) -> Dataset:
    """Get the Lizard dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    data_paths = get_lizard_paths(path, split, download)

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
    path: Union[os.PathLike, str], patch_shape: Tuple[int, int], batch_size: int, split: str, download: bool = False, **kwargs
) -> DataLoader:
    """Get the Lizard dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_lizard_dataset(path, patch_shape, download=download, split=split, **ds_kwargs)
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
