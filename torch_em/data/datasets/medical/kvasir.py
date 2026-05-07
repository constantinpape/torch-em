"""The KVASIR dataset contains annotations for polyp segmentation
in colonoscopy images.

The dataset is located at: https://datasets.simula.no/kvasir-seg/.
This dataset is from the publication https://doi.org/10.1007/978-3-030-37734-2_37.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
CHECKSUM = "03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7"


def get_kvasir_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the KVASIR dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Kvasir-SEG")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "kvasir-seg.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_kvasir_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the KVASIR data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_kvasir_data(path=path, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "images", "*.jpg")))
    gt_paths = sorted(glob(os.path.join(data_dir, "masks", "*.jpg")))

    neu_gt_dir = os.path.join(data_dir, "masks", "preprocessed")
    os.makedirs(neu_gt_dir, exist_ok=True)

    neu_gt_paths = []
    for gt_path in tqdm(gt_paths):
        neu_gt_path = os.path.join(neu_gt_dir, f"{Path(gt_path).stem}.tif")
        neu_gt_paths.append(neu_gt_path)
        if os.path.exists(neu_gt_path):
            continue

        gt = imageio.imread(gt_path)
        gt = np.mean(gt, axis=-1)
        gt = (gt >= 240).astype("uint8")
        imageio.imwrite(neu_gt_path, gt, compression="zlib")

    return image_paths, neu_gt_paths


def get_kvasir_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the KVASIR dataset for polyp segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_kvasir_paths(path, download)

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


def get_kvasir_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the KVASIR dataloader for polyp segmentation.

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
    dataset = get_kvasir_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
