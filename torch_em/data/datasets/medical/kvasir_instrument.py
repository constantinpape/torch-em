"""The Kvasir-Instrument dataset contains annotations for surgical instrument
(e.g. snares, biopsy forceps) segmentation in gastrointestinal endoscopy.

NOTE: This is a different dataset from Kvasir-SEG (polyp segmentation, see `kvasir.py`).

The dataset is located at https://datasets.simula.no/kvasir-instrument/,
mirrored on Kaggle at https://www.kaggle.com/datasets/debeshjha1/kvasirinstrument.
This dataset is from the publication https://doi.org/10.1007/978-3-030-67835-7_19.
Please cite it if you use this dataset for your research.
"""

import os
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


CHECKSUM = "8e12d6c9e232e2f3db90f325ae6e5f81143b5704afcf94d94835c6aa3f1314e1"


def get_kvasir_instrument_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Kvasir-Instrument dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "kvasir-instrument")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name="debeshjha1/kvasirinstrument", download=download)
    zip_path = os.path.join(path, "kvasirinstrument.zip")
    util._check_checksum(zip_path, CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_kvasir_instrument_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Kvasir-Instrument data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_kvasir_instrument_data(path=path, download=download)

    split_file = os.path.join(data_dir, f"{split}.txt")
    with open(split_file) as f:
        image_ids = [line.strip() for line in f if line.strip()]

    neu_gt_dir = os.path.join(data_dir, "masks", "preprocessed")
    os.makedirs(neu_gt_dir, exist_ok=True)

    image_paths, gt_paths = [], []
    for image_id in tqdm(image_ids):
        image_path = os.path.join(data_dir, "images", "images", f"{image_id}.jpg")
        gt_path = os.path.join(data_dir, "masks", "masks", f"{image_id}.png")
        neu_gt_path = os.path.join(neu_gt_dir, f"{image_id}.tif")

        image_paths.append(image_path)
        gt_paths.append(neu_gt_path)
        if os.path.exists(neu_gt_path):
            continue

        gt = imageio.imread(gt_path)
        gt = np.mean(gt, axis=-1)
        gt = (gt >= 240).astype("uint8")
        imageio.imwrite(neu_gt_path, gt, compression="zlib")

    return natsorted(image_paths), natsorted(gt_paths)


def get_kvasir_instrument_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Kvasir-Instrument dataset for surgical instrument segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_kvasir_instrument_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_kvasir_instrument_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Kvasir-Instrument dataloader for surgical instrument segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_kvasir_instrument_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
