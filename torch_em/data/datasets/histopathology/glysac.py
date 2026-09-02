"""The GLySAC dataset contains annotations for nuclei instance segmentation and
classification in H&E stained gastric cancer histopathology images.

The dataset contains 59 image tiles of size 1000x1000 pixels with instance
segmentation masks and cell type annotations. Three cell classes are provided:
lymphocytes, epithelial cells (normal and tumor), and other cells.

NOTE: The dataset is hosted on Google Drive and requires gdown to download.
Install it with: conda install -c conda-forge gdown==4.6.3

The dataset is located at https://drive.google.com/file/d/1g1_xYFWgp3cRLKrlSwD2U5JDjooC0yHp/view
This dataset is from the publication https://doi.org/10.1109/jbhi.2022.3149936.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import List, Literal, Tuple, Union

import h5py
import imageio.v3 as imageio
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


GDRIVE_ID = "1g1_xYFWgp3cRLKrlSwD2U5JDjooC0yHp"
URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"
CHECKSUM = None


def _create_h5_files(data_dir: str, split: str) -> None:
    folder = "Train" if split == "train" else "Test"
    image_dir = os.path.join(data_dir, folder, "Images")
    label_dir = os.path.join(data_dir, folder, "Labels")
    h5_dir = os.path.join(data_dir, "h5", split)
    os.makedirs(h5_dir, exist_ok=True)

    image_paths = natsorted(glob(os.path.join(image_dir, "*.png")))
    for image_path in tqdm(image_paths, desc=f"Preprocessing {split}"):
        fname = os.path.splitext(os.path.basename(image_path))[0]
        h5_path = os.path.join(h5_dir, f"{fname}.h5")
        if os.path.exists(h5_path):
            continue

        label_path = os.path.join(label_dir, f"{fname}.mat")
        raw = imageio.imread(image_path)[..., :3]
        mat = loadmat(label_path)
        inst_map = mat["inst_map"].astype("int32")
        type_map = mat["type_map"].astype("int32")

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw.transpose(2, 0, 1), compression="gzip")
            f.create_dataset("labels/instances", data=inst_map, compression="gzip")
            f.create_dataset("labels/semantic", data=type_map, compression="gzip")


def get_glysac_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the GLySAC dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the data directory.
    """
    data_dir = os.path.join(path, "glysac_dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "glysac_dataset.zip")
    util.download_source_gdrive(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path, path)

    return data_dir


def get_glysac_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    download: bool = False,
) -> List[str]:
    """Get paths to the GLySAC data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the h5 data.
    """
    if split not in ("train", "test"):
        raise ValueError(f"'{split}' is not a valid split. Choose from 'train' or 'test'.")

    data_dir = get_glysac_data(path, download)
    _create_h5_files(data_dir, split)

    h5_paths = natsorted(glob(os.path.join(data_dir, "h5", split, "*.h5")))
    if len(h5_paths) == 0:
        raise RuntimeError(f"No data found for split '{split}'. Check the dataset at {data_dir}.")

    return h5_paths


def get_glysac_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    label_choice: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the GLySAC dataset for gastric nuclei segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'test'.
        label_choice: The type of labels to load. Either 'instances' for instance segmentation
            or 'semantic' for cell type classification (4 classes: other, lymphocyte, epithelial, ambiguous).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if label_choice not in ("instances", "semantic"):
        raise ValueError(f"'{label_choice}' is not a valid label choice. Use 'instances' or 'semantic'.")

    h5_paths = get_glysac_paths(path, split, download)

    if label_choice == "instances":
        kwargs, _ = util.add_instance_label_transform(kwargs, add_binary_target=True)
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=h5_paths,
        raw_key="raw",
        label_paths=h5_paths,
        label_key=f"labels/{label_choice}",
        patch_shape=patch_shape,
        with_channels=True,
        ndim=2,
        **kwargs,
    )


def get_glysac_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    label_choice: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the GLySAC dataloader for gastric nuclei segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'test'.
        label_choice: The type of labels to load. Either 'instances' for instance segmentation
            or 'semantic' for cell type classification (4 classes: other, lymphocyte, epithelial, ambiguous).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_glysac_dataset(path, patch_shape, split, label_choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
