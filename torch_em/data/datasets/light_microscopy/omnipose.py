"""The OmniPose dataset contains phase-contrast and fluorescence microscopy images
and annotations for bacteria segmentation and brightfield microscopy images and
annotations for worm segmentation.

This dataset is described in the publication https://doi.org/10.1038/s41592-022-01639-4.
Please cite it if you use this dataset in your research.
"""


import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, Optional, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://files.osf.io/v1/resources/xmury/providers/osfstorage/62f56c035775130690f25481/?zip="

# NOTE: the checksums are not reliable from the osf project downloads.
# CHECKSUM = "7ae943ff5003b085a4cde7337bd9c69988b034cfe1a6d3f252b5268f1f4c0af7"
CHECKSUM = None

DATA_CHOICES = ["bact_fluor", "bact_phase", "worm", "worm_high_res"]


def get_omnipose_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the OmniPose dataset.

    Args:
        path: Filepath to the folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Return:
        The filepath where the data is downloaded.
    """
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "datasets.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def get_omnipose_paths(
    path: Union[os.PathLike, str],
    split: str,
    data_choice: Optional[Union[str, List[str]]] = None,
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the OmniPose data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train' or 'test'.
        data_choice: The choice of specific data.
            Either 'bact_fluor', 'bact_phase', 'worm' or 'worm_high_res'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_omnipose_data(path=path, download=download)

    if split not in ["train", "test"]:
        raise ValueError(f"'{split}' is not a valid split.")

    if data_choice is None:
        data_choice = DATA_CHOICES
    else:
        if not isinstance(data_choice, list):
            data_choice = [data_choice]

    all_image_paths, all_gt_paths = [], []
    for _chosen_data in data_choice:
        if _chosen_data not in DATA_CHOICES:
            raise ValueError(f"'{_chosen_data}' is not a valid choice of data.")

        if _chosen_data.startswith("bact"):
            base_dir = os.path.join(data_dir, _chosen_data, f"{split}_sorted", "*")
            gt_paths = glob(os.path.join(base_dir, "*_masks.tif"))
            image_paths = glob(os.path.join(base_dir, "*.tif"))

        else:
            base_dir = os.path.join(data_dir, _chosen_data, split)
            gt_paths = glob(os.path.join(base_dir, "*_masks.*"))
            image_paths = glob(os.path.join(base_dir, "*"))

        for _path in image_paths.copy():
            # NOTE: Removing the masks and flows from the image paths.
            if _path.endswith("_masks.tif") or _path.endswith("_masks.png") or _path.endswith("_flows.tif"):
                image_paths.remove(_path)

        all_image_paths.extend(natsorted(image_paths))
        all_gt_paths.extend(natsorted(gt_paths))

    return all_image_paths, all_gt_paths


def get_omnipose_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    data_choice: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the OmniPose dataset for segmenting bacteria and worms in microscopy images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train' or 'test'.
        data_choice: The choice of specific data.
            Either 'bact_fluor', 'bact_phase', 'worm' or 'worm_high_res'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_omnipose_paths(path, split, data_choice, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_omnipose_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: Literal["train", "test"],
    data_choice: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the OmniPose dataloader for segmenting bacteria and worms in microscopy images.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The data split to use. Either 'train' or 'test'.
        data_choice: The choice of specific data.
            Either 'bact_fluor', 'bact_phase', 'worm' or 'worm_high_res'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_omnipose_dataset(
        path=path, patch_shape=patch_shape, split=split, data_choice=data_choice, download=download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
