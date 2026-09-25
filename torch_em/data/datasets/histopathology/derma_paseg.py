"""DERMA-PASeg contains annotations for semantic segmentation of skin tissue layers in
brightfield whole-slide histopathology images.

The dataset provides 32 skin biopsy sections, each imaged unstained, chemically PAS-stained,
and as a GAN-generated "virtually stained" counterpart, together with a semantic segmentation
mask for 5 classes: background and the Dermis, Epidermis, Keratin and Dermal-Epidermal
Junction layers. The source ships the mask as an RGB image without a color legend, so this
module maps its 5 colors to label ids 0-4 by sorting the RGB triplets, without claiming
which id corresponds to which named layer.

NOTE: The chemically-stained image is missing for one training sample; that pair is skipped
when `stain="chemically_stained"`.

The dataset is located at https://data.mendeley.com/datasets/w8vxx8yz55/1 and licensed under
CC BY 4.0.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import List, Literal, Tuple, Union

import h5py
import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://data.mendeley.com/public-api/zip/w8vxx8yz55/download/1"
CHECKSUM = "e42604f64f2e047c8dac1a7ae23650aee91cd01dedcd65b6cec861eb0af87a62"

# Sorted so that background (0, 0, 0) maps to label id 0.
LABEL_COLORS = [(0, 0, 0), (112, 48, 160), (190, 255, 0), (224, 224, 224), (255, 172, 255)]

STAIN_FOLDERS = {"unstained": "Unstained", "chemically_stained": "C Stained", "virtually_stained": "V Stained"}


def _find_raw_path(data_dir, folder, stain, base_name):
    stain_dir = os.path.join(data_dir, STAIN_FOLDERS[stain], folder)
    if stain == "unstained":
        candidate = os.path.join(stain_dir, f"{base_name}.jpg")
        return candidate if os.path.exists(candidate) else None
    elif stain == "chemically_stained":
        candidate = os.path.join(stain_dir, f"{base_name}-PAS.jpg")
        return candidate if os.path.exists(candidate) else None
    else:  # The virtually stained images carry an inconsistent 'blended_final' / 'blended_test' suffix.
        matches = glob(os.path.join(stain_dir, f"{base_name}.blended_*.jpg"))
        return matches[0] if matches else None


def _create_h5_files(data_dir, split, stain):
    folder = "Train" if split == "train" else "Test"
    h5_dir = os.path.join(data_dir, "h5", stain, split)
    os.makedirs(h5_dir, exist_ok=True)

    mask_paths = natsorted(glob(os.path.join(data_dir, "Masks", folder, "*.png")))
    for mask_path in tqdm(mask_paths, desc=f"Preprocessing {split} ({stain})"):
        base_name = os.path.splitext(os.path.basename(mask_path))[0]
        h5_path = os.path.join(h5_dir, f"{base_name}.h5")
        if os.path.exists(h5_path):
            continue

        raw_path = _find_raw_path(data_dir, folder, stain, base_name)
        if raw_path is None:
            continue

        raw = imageio.imread(raw_path)[..., :3]
        mask = imageio.imread(mask_path)[..., :3]
        labels = np.zeros(mask.shape[:2], dtype="uint8")
        for label_id, color in enumerate(LABEL_COLORS):
            labels[np.all(mask == color, axis=-1)] = label_id

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw.transpose(2, 0, 1), compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_derma_paseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the DERMA-PASeg dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the data directory.
    """
    data_dir = os.path.join(path, "DERMA-PASeg", "DERMA-PASeg")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "derma_paseg.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_derma_paseg_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    stain: Literal["unstained", "chemically_stained", "virtually_stained"] = "unstained",
    download: bool = False,
) -> List[str]:
    """Get paths to the DERMA-PASeg data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train' or 'test'.
        stain: The image variant to use as raw data. One of 'unstained', 'chemically_stained'
            or 'virtually_stained'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the h5 data.
    """
    if split not in ("train", "test"):
        raise ValueError(f"'{split}' is not a valid split. Choose from 'train' or 'test'.")
    if stain not in STAIN_FOLDERS:
        raise ValueError(f"'{stain}' is not a valid stain. Choose from {list(STAIN_FOLDERS.keys())}.")

    data_dir = get_derma_paseg_data(path, download)
    _create_h5_files(data_dir, split, stain)

    h5_paths = natsorted(glob(os.path.join(data_dir, "h5", stain, split, "*.h5")))
    if len(h5_paths) == 0:
        raise RuntimeError(f"No data found for split '{split}' and stain '{stain}'. Check the dataset at {data_dir}.")

    return h5_paths


def get_derma_paseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    stain: Literal["unstained", "chemically_stained", "virtually_stained"] = "unstained",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the DERMA-PASeg dataset for skin tissue layer segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'test'.
        stain: The image variant to use as raw data. One of 'unstained', 'chemically_stained'
            or 'virtually_stained'.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    h5_paths = get_derma_paseg_paths(path, split, stain, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=h5_paths,
        raw_key="raw",
        label_paths=h5_paths,
        label_key="labels",
        patch_shape=patch_shape,
        with_channels=True,
        ndim=2,
        **kwargs,
    )


def get_derma_paseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    stain: Literal["unstained", "chemically_stained", "virtually_stained"] = "unstained",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DERMA-PASeg dataloader for skin tissue layer segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'test'.
        stain: The image variant to use as raw data. One of 'unstained', 'chemically_stained'
            or 'virtually_stained'.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_derma_paseg_dataset(path, patch_shape, split, stain, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
