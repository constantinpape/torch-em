"""The DeepSeas dataset contains annotations for cell segmentation in
phase-contrast microscopy images of stem cells, bronchial cells and muscle cells.

NOTE: Please download the dataset manually from https://drive.google.com/drive/folders/18odgkzafW8stHkzME_s7Es-ue7odVAc5?usp=sharing.
The original data is located at: https://drive.google.com/drive/folders/13RhhBAetSWkjySyhJcDqj_FaO09hxkhO?usp=sharing.
The tracking data is located at: https://drive.google.com/drive/folders/10LWey85fgHgFj_myIr1CYSOviD4SleE4?usp=sharing.

The dataset is located at https://deepseas.org/datasets/.
The codebase for this dataset is located at https://github.com/abzargar/DeepSea.
Please cite them if you use this dataset for your research.
"""  # noqa

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://drive.google.com/drive/folders/18odgkzafW8stHkzME_s7Es-ue7odVAc5?usp=sharing"


def get_deepseas_data(path: Union[os.PathLike, str], split: Literal['train', 'test'], download: bool = False) -> str:
    """Obtain the DeepSeas dataset.

    NOTE: You need to manually download the 'segmentation_dataset' from the link:
    - https://drive.google.com/drive/folders/18odgkzafW8stHkzME_s7Es-ue7odVAc5?usp=sharing.

    Args:
        path: Filepath to a folder where the downloaded data will be stored.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is manually downloaded for further processing.
    """
    if split not in ["train", "test"]:
        raise ValueError(f"'{split}' is not a valid split choice.")

    data_dir = os.path.join(path, "segmentation_dataset", split)
    if os.path.exists(data_dir):
        return data_dir

    zip_paths = glob(os.path.join(path, "*.zip"))
    if len(zip_paths) == 0 or download:
        raise NotImplementedError(
            "Automatic download for DeepSeas data is not supported at the moment. "
            f"Please download the 'segmentation_dataset' from {URL} and place the zip files at {path}."
        )

    for zip_path in zip_paths:
        util.unzip(zip_path=zip_path, dst=path, remove=False)

    return data_dir


def get_deepseas_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'test'], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the DeepSeas data.

    Args:
        path: Filepath to a folder where the downloaded data will be stored.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_deepseas_data(path, split, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "images", "*.png")))
    label_paths = natsorted(glob(os.path.join(data_dir, "masks", "*.png")))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_deepseas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DeepSeas dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be stored.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_deepseas_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        with_channels=True,
        **kwargs
    )


def get_deepseas_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DeepSeas dataloader for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be stored.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_deepseas_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
