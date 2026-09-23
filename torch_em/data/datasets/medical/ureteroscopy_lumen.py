"""The Ureteroscopy Lumen Segmentation dataset contains annotations for ureteral lumen segmentation
in endoscopic ureteroscopy images.

The dataset provides 'train' and 'val' splits with images paired one-to-one with binary lumen masks,
and a 'test' split organized per-patient ('test_01', 'test_02', 'test_03'), which this loader merges
into a single flat split. The masks ship as near-binary RGB PNGs with a small amount of JPEG-style
compression noise, so this loader binarizes them (foreground where any channel exceeds half intensity)
and stores them as single-channel tif files during preprocessing.

NOTE: The Zenodo record description reports two inconsistent counts, "1,754 images from 23 patients"
and "2,187 images with masks". This loader does not rely on either advertised count and instead
discovers the image-mask pairs on disk, which totals 2,181 pairs (798 train, 417 val, 966 test).

The dataset is located at https://zenodo.org/records/10066606, released under a CC-BY-4.0 license.

This dataset is from the publication https://doi.org/10.1109/ICPR48806.2021.9412209.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/10066606/files/lumen_dataset.zip"
CHECKSUM = "e2a99c1bd59453d0bf7eed3fb916c0ea6526f4d9b65851d9759f919362effdc8"

SPLITS = ["train", "val", "test"]


def _preprocess_data(data_dir):
    preprocessed_dir = os.path.join(data_dir, "preprocessed")
    if os.path.exists(preprocessed_dir):
        return preprocessed_dir

    raw_dir = os.path.join(data_dir, "lumen_dataset")
    for split in SPLITS:
        image_dir = os.path.join(preprocessed_dir, split, "images")
        label_dir = os.path.join(preprocessed_dir, split, "labels")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        if split == "test":
            source_dirs = natsorted(glob(os.path.join(raw_dir, "test", "test_*")))
        else:
            source_dirs = [os.path.join(raw_dir, split)]

        for source_dir in tqdm(source_dirs, desc=f"Preprocessing '{split}' split"):
            label_paths = natsorted(glob(os.path.join(source_dir, "label", "*.png")))
            for label_path in label_paths:
                fname = os.path.basename(label_path)
                image_path = os.path.join(source_dir, "image", fname)
                if not os.path.exists(image_path):
                    continue

                label = imageio.imread(label_path)
                label = (np.any(label > 127, axis=-1)).astype("uint8")
                image = imageio.imread(image_path)

                out_name = os.path.splitext(f"{os.path.basename(source_dir)}_{fname}")[0] + ".tif"
                imageio.imwrite(os.path.join(image_dir, out_name), image)
                imageio.imwrite(os.path.join(label_dir, out_name), label)

    return preprocessed_dir


def get_ureteroscopy_lumen_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Ureteroscopy Lumen Segmentation dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    data_dir = os.path.join(path, "lumen_dataset")
    if os.path.exists(data_dir):
        return _preprocess_data(path)

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "lumen_dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return _preprocess_data(path)


def get_ureteroscopy_lumen_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Ureteroscopy Lumen Segmentation data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose one of {SPLITS}.")

    preprocessed_dir = get_ureteroscopy_lumen_data(path, download)

    raw_paths = natsorted(glob(os.path.join(preprocessed_dir, split, "images", "*.tif")))
    label_paths = natsorted(glob(os.path.join(preprocessed_dir, split, "labels", "*.tif")))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_ureteroscopy_lumen_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Ureteroscopy Lumen Segmentation dataset for lumen segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_ureteroscopy_lumen_paths(path, split, download)

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


def get_ureteroscopy_lumen_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Ureteroscopy Lumen Segmentation dataloader for lumen segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ureteroscopy_lumen_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
