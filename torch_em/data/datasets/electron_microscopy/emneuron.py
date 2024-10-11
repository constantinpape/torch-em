"""EMNeuron is a dataset for neuron segmentation in EM.
It contains multiple annotated volumes from 14 domain sources.

The dataset is hosted at https://huggingface.co/datasets/yanchaoz/EMNeuron.
The dataset is published in https://papers.miccai.org/miccai-2024/677-Paper0518.html.
Please cite this publication if you use the dataset in your research.
"""

import os
import shutil
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def _clean_redundant_files(path):
    # The "InDistribution" directory is redundant.
    target_dir = os.path.join(path, "valid", "InDistribution", "InDistribution")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)


def get_emneuron_data(path: Union[os.PathLike, str], split: Literal['train', 'val'], download: bool = False):
    """Get the EMNeuron data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split of the data to be used for training.
        download: Whether to download the data if it is not present.
    """
    if download:
        raise ValueError()

    os.makedirs(path, exist_ok=True)

    if split == "train":
        rar_path = os.path.join(path, "labeled.rar")
    elif split == "val":
        rar_path = os.path.join(path, "valid.rar")
    else:
        raise ValueError(f"'{split}' is not a valid split. Please choose either 'train' or 'val'.")

    if os.path.exists(os.path.splitext(rar_path)[0]):
        return

    util.unzip_rarfile(rar_path=rar_path, dst=path, remove=False, use_rarfile=False)

    _clean_redundant_files(path)


def get_emneuron_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'val'], download: bool = False
) -> List[str]:
    """Get paths to the EMNeuron data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split of the data to be used for training.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the stored data.
    """
    get_emneuron_data(path, split, download)
    if split == "train":
        label_paths = natsorted(glob(os.path.join(path, "labeled", "*", "*_MaskIns.tif")))
        raw_paths = [os.path.join(os.path.dirname(p), os.path.basename(p).replace("_MaskIns", "")) for p in label_paths]

    else:  # 'val' split
        raw_paths = natsorted(glob(os.path.join(path, "valid", "*", "*", "raw.tif")))
        label_paths = [
            os.path.join(os.path.dirname(p), "label_0.tif")
            if os.path.exists(os.path.join(os.path.dirname(p), "label_0.tif"))
            else os.path.join(os.path.dirname(p), "label.tif") for p in raw_paths
        ]

    assert len(raw_paths) == len(label_paths)
    return raw_paths, label_paths


def get_emneuron_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val'],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the dataset for EMNeuron dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The split of the data to be used for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_emneuron_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_emneuron_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val'],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the dataloader for EMNeuron dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The split of the data to be used for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_emneuron_dataset(path=path, patch_shape=patch_shape, split=split, download=download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
