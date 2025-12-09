"""The U20S dataset contains annotations for nucleus segmentation in
fluoroscence microscopy images of U20S cells.

The dataset is hosted at https://bbbc.broadinstitute.org/BBBC039.
This dataset is available as a BBBC collection, published by https://www.nature.com/articles/nmeth.2083.
Please cite it if you use this dataset for your research.
"""

import os
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import List, Union, Tuple

import imageio.v3 as imageio
from skimage.measure import label as connected_components

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "images": "https://data.broadinstitute.org/bbbc/BBBC039/images.zip",
    "masks": "https://data.broadinstitute.org/bbbc/BBBC039/masks.zip"
}

CHECKSUMS = {
    "images": "6f30a5d4fe38c928ded972704f085975f8dc0d65d9aa366df00e5a9d449fddd7",
    "masks": "f9e6043d8ca56344a4886f96a700d804d6ee982f31e2b2cd3194af2a053c2710"
}


def _process_masks(path):
    label_dir = os.path.join(path, "labels")
    os.makedirs(label_dir)

    for p in tqdm(glob(os.path.join(path, "masks", "*.png")), desc="Processing masks"):
        curr_mask = imageio.imread(p)

        assert curr_mask.ndim == 3 and curr_mask.shape[-1] == 4  # Making the obvious assumption here.

        # Choose the first channel and run cc.
        curr_mask = connected_components(curr_mask[:, :, 0])

        # Store labels as tif now.
        imageio.imwrite(os.path.join(label_dir, f"{Path(p).stem}.tif"), curr_mask, compression="zlib")

    # Remove the mask directory and random MAC cache files now.
    shutil.rmtree(os.path.join(path, "masks"))
    shutil.rmtree(os.path.join(path, "__MACOSX"))


def get_u20s_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the U20S dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        The path where the dataset is downloaded for further processing.
    """
    label_dir = os.path.join(path, "labels")
    if os.path.exists(label_dir):
        return path

    os.makedirs(path, exist_ok=True)

    # Download the image and labels
    for name, url in URLS.items():
        zip_path = os.path.join(path, f"{name}.zip")
        util.download_source(path=zip_path, url=url, download=download, checksum=CHECKSUMS[name])
        util.unzip(zip_path, dst=path)

    # Postprocess masks
    _process_masks(path)

    return path


def get_u20s_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Usiigaci data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_u20s_data(path, download)

    image_paths = natsorted(glob(os.path.join(data_dir, "images", "*.tif")))
    label_paths = natsorted(glob(os.path.join(data_dir, "labels", "*.tif")))

    return image_paths, label_paths


def get_u20s_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the U20S dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    image_paths, label_paths = get_u20s_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        ndim=2,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_u20s_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the U20S dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_u20s_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
