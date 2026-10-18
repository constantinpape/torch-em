"""The SNEMI Mito dataset contains annotations for mitochondria segmentation in electron microscopy (EM)
images of mouse neocortex.

It provides mitochondria labels for the sections of the SNEMI3D challenge volume, which comes from the publication
https://doi.org/10.1016/j.cell.2015.06.054. The data consists of 100 images (1024 x 1024 pixels, 8-bit, CLAHE contrast
enhanced) that are split into 50 train, 20 validation and 30 test images.

The data is available at https://github.com/sumiya-kuroda/EMsegmentation-mito/tree/master/dataset/SNEMI3D_mito.
The repository does not state a license. The mitochondria labels were prepared by members of the Kubota lab
(see the repository for details). Please cite the SNEMI3D / Kasthuri et al. publication and the repository
if you use this dataset in your research.
"""

import os
import shutil
from glob import glob
from typing import List, Tuple, Union

from natsort import natsorted
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


# The archive is pinned to a commit, so that the checksum does not change with new commits to the repository.
URL = "https://github.com/sumiya-kuroda/EMsegmentation-mito/archive/44134f6dc27b1ad7f04293398e482fec854bb658.zip"
CHECKSUM = "1d560ad05d0faab8e6fedd9347131c2a0f395dcdba079f8699a66c3e39d80a25"
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def get_snemi_mito_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the SNEMI Mito dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the data.
    """
    data_dir = os.path.join(path, "SNEMI3D_mito")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "snemi_mito.zip")
    util.download_source(zip_path, URL, download, checksum=CHECKSUM)

    # The archive contains the whole repository, we only keep the SNEMI3D_mito folder.
    tmp_dir = os.path.join(path, "tmp")
    util.unzip(zip_path, tmp_dir, remove=True)
    extracted = glob(os.path.join(tmp_dir, "*", "dataset", "SNEMI3D_mito"))
    assert len(extracted) == 1, "Unexpected structure of the downloaded archive."
    shutil.move(extracted[0], data_dir)
    shutil.rmtree(tmp_dir)

    return data_dir


def get_snemi_mito_paths(
    path: Union[os.PathLike, str], split: str, download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the SNEMI Mito data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Please choose from {list(SPLITS.keys())}.")

    data_dir = get_snemi_mito_data(path, download)
    raw_paths = natsorted(glob(os.path.join(data_dir, SPLITS[split], "images", "*.png")))
    label_paths = natsorted(glob(os.path.join(data_dir, SPLITS[split], "labels", "*.png")))
    assert len(raw_paths) > 0 and len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def _binary_labels(labels):
    return (labels > 0).astype("uint8")


def get_snemi_mito_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: str,
    binary: bool = True,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SNEMI Mito dataset for mitochondria segmentation in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        binary: Whether to return a binary foreground mask. Otherwise the mask values are returned as they are.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_snemi_mito_paths(path, split, download)

    if binary:
        kwargs = util.update_kwargs(kwargs, "label_transform", _binary_labels)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_snemi_mito_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: str,
    binary: bool = True,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SNEMI Mito dataloader for mitochondria segmentation in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        binary: Whether to return a binary foreground mask. Otherwise the mask values are returned as they are.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_snemi_mito_dataset(path, patch_shape, split, binary, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
