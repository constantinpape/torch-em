"""The cytoDArk0 dataset contains cell annotations for Nissl-stained histological images of mammalian brain.

NOTE: The dataset contains instance segmentation annotations of all types of neuron and glia cells.
In addition, it contains semantic segmentation annotations for foreground (cells) vs background vs boundary between
touching and closely positioned cells (four-classes in total).

The original dataset is located at https://zenodo.org/records/13694738.
The dataset is from the publication https://www.sciencedirect.com/science/article/pii/S0010482525013708.
Please cite it if you use this dataset for your research.
"""

import os
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Union, Tuple, Literal, List, Optional

import pandas as pd
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/13694738/files/cytoDArk0.zip"
CHECKSUM = "ce4b05675aa5057e277c8d4ab74524307e2402a3703f6bd80643b93ca9b70ff8"


def _preprocess_images(path, data_dir):
    import h5py

    def _process_per_magnification(mag):
        # Let's sort one magnification images first.
        if mag == "20x":
            base_dir = os.path.join(data_dir, "20x", "1024x1024")
        elif mag == "40x":
            base_dir = os.path.join(data_dir, "40x", "2048x2048")
        else:
            raise ValueError

        preprocessed_dir = os.path.join(path, "preprocessed", mag)
        os.makedirs(preprocessed_dir, exist_ok=True)

        # 1. Load each image and corresponding labels
        for image_path in tqdm(glob(os.path.join(base_dir, "image", "*.png")), desc=f"Preprocess {mag} images"):
            image_name = Path(image_path).stem

            image = imageio.imread(image_path)
            instances = imageio.imread(os.path.join(base_dir, "label", f"{image_name}.tiff"))
            semantics = imageio.imread(os.path.join(base_dir, "graymask4", f"{image_name}.png"))

            with h5py.File(os.path.join(preprocessed_dir, f"{image_name}.h5"), "w") as f:
                f.create_dataset("raw", data=image, compression="gzip")
                f.create_dataset("labels/instances", data=instances, compression="gzip")
                f.create_dataset("labels/semantic/pixels_classification", data=semantics, compression="gzip")

        # Next, let's sort them in split folders.
        # 1. Load the file with fold information.
        fold = pd.read_csv(os.path.join(base_dir, "folds.csv"))

        # 2. Make split folders, find files and drop them.
        train_paths, val_paths, test_paths = (fold.loc[fold["fold"] == i, "img_id"].tolist() for i in range(3))

        train_paths = [os.path.join(preprocessed_dir, f"{p}.h5") for p in train_paths]
        val_paths = [os.path.join(preprocessed_dir, f"{p}.h5") for p in val_paths]
        test_paths = [os.path.join(preprocessed_dir, f"{p}.h5") for p in test_paths]

        # Move them to their own split folders.
        def _move_files(split, paths):
            assert split in ["train", "val", "test"]

            trg_dir = os.path.join(preprocessed_dir, split)
            os.makedirs(trg_dir, exist_ok=True)
            [shutil.move(p, os.path.join(trg_dir, os.path.basename(p))) for p in paths]

        _move_files("train", train_paths)
        _move_files("val", val_paths)
        _move_files("test", test_paths)

    _process_per_magnification("20x")
    _process_per_magnification("40x")

    # Finally, remove all other files because we don't care about them anymore.
    shutil.rmtree(data_dir)


def get_cytodark0_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the cytoDArk0 dataset.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where dataset is downloaded for further processing.
    """
    data_dir = os.path.join(path, "preprocessed")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "cytoDArk0.zip")
    util.download_source(zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path, path)

    _preprocess_images(path, os.path.join(path, "cytoDArk0"))

    return data_dir


def get_cytodark0_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    magnification: Optional[Literal["20x", "40x"]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the cytoDArk0 data.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        magnification: The choice of magnification, by default returns all images across all magnification,
            i.e. '20x' and '40x'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    data_dir = get_cytodark0_data(path, download)

    assert split in ["train", "val", "test"], split
    if magnification:
        assert magnification in ["20x", "40x"], magnification

    input_paths = glob(os.path.join(data_dir, magnification, split, "*.h5"))
    return input_paths


def get_cytodark0_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    magnification: Optional[Literal["20x", "40x"]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the cytoDArk0 dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        magnification: The choice of magnification, by default returns all images across all magnification,
            i.e. '20x' and '40x'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    input_paths = get_cytodark0_paths(path, split, magnification, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=input_paths,
        raw_key="raw",
        label_paths=input_paths,
        label_key="labels/instances",
        patch_shape=patch_shape,
        ndim=2,
        **kwargs
    )


def get_cytodark0_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    magnification: Optional[Literal["20x", "40x"]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the cytoDArk0 dataloader for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        batch_size: The batch size for training
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        magnification: The choice of magnification, by default returns all images across all magnification,
            i.e. '20x' and '40x'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cytodark0_dataset(path, patch_shape, split, magnification, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
