"""
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components

import torch_em

from torch.utils.data import Dataset, DataLoader

from .. import util


URL = "https://zenodo.org/records/7647846/files/IHC_TMA_dataset.zip"
CHECKSUM = "9dcc1c94b5d8af5383d3c91141617b1621904ee9bd6f69d2223e7f4363cc80d9"


def _preprocess_data(data_dir):
    preprocessed_label_dir = os.path.join(data_dir, "preprocessed_labels")
    os.makedirs(preprocessed_label_dir, exist_ok=True)

    label_paths = glob(os.path.join(data_dir, "masks", "*.npy"))
    for lpath in tqdm(label_paths, desc="Preprocessing labels"):
        fname = Path(lpath).stem
        larray = np.load(lpath)
        labels = larray[0] + larray[1]
        labels = connected_components(labels)

        imageio.imwrite(os.path.join(preprocessed_label_dir, f"{fname}.tif"), labels, compression="zlib")


def get_srsanet_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """
    """
    data_dir = os.path.join(path, "IHC_TMA_dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "IHC_TMA_dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    _preprocess_data(data_dir)

    return data_dir


def get_srsanet_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'val', 'test'],
    download: bool = False
) -> Tuple[List[int], List[int]]:
    """
    """
    data_dir = get_srsanet_data(path, download)

    if split == "train":
        dname = "fold1"
    elif split == "val":
        dname = "fold2"
    elif split == "test":
        dname = "fold3"
    else:
        raise ValueError(f"'{split}' is not a valid split choice.")

    raw_paths = natsorted(glob(os.path.join(data_dir, "images", f"{dname}_*.png")))
    label_paths = natsorted(glob(os.path.join(data_dir, "preprocessed_labels", f"{dname}_*.tif")))

    return raw_paths, label_paths


def get_srsanet_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    download: bool = False,
    **kwargs
) -> Dataset:
    """
    """
    raw_paths, label_paths = get_srsanet_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_srsanet_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_srsanet_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
