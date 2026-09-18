"""The EDD2020 dataset contains annotations for multi-class disease segmentation in
gastrointestinal endoscopy images.

The dataset consists of 386 endoscopy frames from 5 different institutions and multiple
GI organs (esophagus, stomach, colon), collected with white light, narrow-band imaging and
chromoendoscopy. Each image is annotated with pixel-level masks for up to 5 disease classes:
non-dysplastic Barrett's esophagus (BE), suspicious lesions, high-grade dysplasia (HGD),
cancer, and polyp. The official challenge data is gated behind manual registration on
https://edd2020.grand-challenge.org, so we instead use the openly mirrored copy at
https://www.kaggle.com/datasets/orvile/edd2020-endoscopy-detection-and-segmentation, which
matches the official release (386 images, same organizers and class structure). NOTE: the
original data release states the license as CC BY-NC-SA 4.0, while the Kaggle mirror lists it
as CC BY 4.0; please check the current license terms before using this data.

This dataset is from the publication https://doi.org/10.1016/j.media.2021.102002.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


KAGGLE_DATASET_NAME = "orvile/edd2020-endoscopy-detection-and-segmentation"

CLASS_NAMES = ["BE", "suspicious", "HGD", "cancer", "polyp"]


def get_edd2020_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the EDD2020 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "EDD2020")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name=KAGGLE_DATASET_NAME, download=download)

    zip_path = os.path.join(path, "edd2020-endoscopy-detection-and-segmentation.zip")
    util.unzip(zip_path=zip_path, dst=path)

    if not os.path.exists(data_dir):
        raise RuntimeError(f"The dataset could not be found at '{path}' after extraction.")

    return data_dir


def get_edd2020_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the EDD2020 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_edd2020_data(path=path, download=download)

    image_paths = natsorted(glob(os.path.join(data_dir, "originalImages", "*.jpg")))
    if len(image_paths) == 0:
        raise RuntimeError("Something went wrong with fetching the image paths.")

    neu_gt_dir = os.path.join(data_dir, "masks", "preprocessed")
    os.makedirs(neu_gt_dir, exist_ok=True)

    gt_paths = []
    for image_path in tqdm(image_paths, desc="Preprocessing labels"):
        image_id = Path(image_path).stem
        gt_path = os.path.join(neu_gt_dir, f"{image_id}.tif")
        gt_paths.append(gt_path)
        if os.path.exists(gt_path):
            continue

        image = imageio.imread(image_path)
        label = np.zeros(image.shape[:2], dtype="uint8")
        for class_id, class_name in enumerate(CLASS_NAMES, start=1):
            mask_path = os.path.join(data_dir, "masks", f"{image_id}_{class_name}.tif")
            if not os.path.exists(mask_path):
                continue
            mask = imageio.imread(mask_path)
            label[mask > 0] = class_id

        imageio.imwrite(gt_path, label, compression="zlib")

    return image_paths, gt_paths


def get_edd2020_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the EDD2020 dataset for multi-class disease segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_edd2020_paths(path, download)

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


def get_edd2020_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the EDD2020 dataloader for multi-class disease segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_edd2020_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
