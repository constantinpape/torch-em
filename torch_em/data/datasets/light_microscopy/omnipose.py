"""The OmniPose dataset contains phase-contrast and fluorescence microscopy images
and annotations for bacteria segmentation and brightfield microscopy images and
annotations for worm segmentation.

This dataset is described in the publication https://doi.org/10.1038/s41592-022-01639-4.
Please cite it if you use this dataset in your research.
"""


import os
from typing import Union, Tuple, Literal, Optional

from .. import util

import torch_em


URL = "https://files.osf.io/v1/resources/xmury/providers/osfstorage/62f56c035775130690f25481/?zip="

# NOTE: the checksums are not reliable from the osf project downloads. 
# CHECKSUM = "7ae943ff5003b085a4cde7337bd9c69988b034cfe1a6d3f252b5268f1f4c0af7"
CHECKSUM = None


def get_omnipose_data(
    path: Union[os.PathLike, str],
    download: bool = False,
):
    """Download the OmniPose dataset.

    Args:
        path: Filepath to the folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Return:
        The path to the parent data directory of the respective choice of data.
    """
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "datasets.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def _get_omnipose_paths(path, split, data_choice, download):
    image_paths, gt_paths = ...
    return image_paths, gt_paths


def get_omnipose_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    data_choice: Optional[Literal["bact_fluor", "bact_phase", "worm", "worm_high_res"]] = None,
    download: bool = False,
    **kwargs
):
    """Get the OmniPose dataset for segmenting bacteria and worms in microscopy images.

    Args:
        path: Filepathg to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.
        data_choice: The data choice
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = _get_omnipose_paths(path, split, data_choice, download)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )
    return dataset


def get_omnipose_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: Literal["train", "test"],
    data_choice: Optional[Literal["bact_fluor", "bact_phase", "worm", "worm_high_res"]] = None,
    download: bool = False,
    **kwargs
):
    """Get the OmniPose dataloader for segmenting bacteria and worms in microscopy images.

    Args:
        path: Filepathg to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The data split to use. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.
        data_choice: The data choice
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_omnipose_dataset(
        path=path, patch_shape=patch_shape, split=split, data_choice=data_choice, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
