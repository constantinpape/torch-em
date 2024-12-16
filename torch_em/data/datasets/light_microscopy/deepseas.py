"""The DeepSeas dataset contains annotations for cell segmentation in
phase-contrast microscopy images of stem cells, bronchial cells and muscle cells.

The dataset is located at https://deepseas.org/datasets/.
The codebase for this dataset is located at https://github.com/abzargar/DeepSea.
Please cite them if you use this dataset for your research.
"""

import os
from typing import Union, Tuple

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = {
    "original": "https://drive.google.com/drive/folders/13RhhBAetSWkjySyhJcDqj_FaO09hxkhO?usp=sharing",
    "segmentation": "https://drive.google.com/drive/folders/18odgkzafW8stHkzME_s7Es-ue7odVAc5?usp=sharing",
    "tracking": "https://drive.google.com/drive/folders/10LWey85fgHgFj_myIr1CYSOviD4SleE4?usp=sharing",
}


def get_deepseas_data(path: Union[os.PathLike, str], choice, download: bool = False):
    """
    """
    data_dir = os.path.join(path, choice)
    if os.path.exists(data_dir):
        return data_dir

    print("'DeepSeas' is a large dataset. Downloading all files might take a while.")
    util.download_source_gdrive(
        path=os.path.join(path, choice),
        url=URL[choice],
        download=download,
        download_type="folder",
        quiet=False,
    )

    return data_dir


def get_deepseas_paths(path: Union[os.PathLike, str], choice, download: bool = False):
    """
    """
    data_dir = get_deepseas_data(path, choice, download)

    breakpoint()

    raw_paths = ...
    label_paths = ...

    return raw_paths, label_paths


def get_deepseas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    choice,
    download: bool = False,
    **kwargs
) -> Dataset:
    """
    """
    raw_paths, label_paths = get_deepseas_paths(path, choice, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs
    )


def get_deepseas_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    choice,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_deepseas_dataset(path, patch_shape, choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
