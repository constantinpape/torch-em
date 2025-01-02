"""The BitDepth NucSeg dataset contains annotations for nucleus segmentation
in DAPI stained fluorescence microscopy images.

The dataset is located at https://github.com/masih4/BitDepth_NucSeg/
This dataset is from the publication https://doi.org/10.3390/diagnostics11060967.
Please cite it if you use this dataset in your research.
"""

import os
import shutil
import subprocess
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Optional, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://github.com/masih4/BitDepth_NucSeg"


def _remove_other_files(path):
    "Remove other files from the git repository"
    all_files = glob(os.path.join(path, "*"))
    all_files.extend(glob(os.path.join(path, ".*")))
    for _file in all_files:
        if os.path.basename(_file) == "data":
            continue

        if os.path.isdir(_file):
            shutil.rmtree(_file)
        else:
            os.remove(_file)


def get_bitdepth_nucseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BitDepth NucSeg dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the downloaded data.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    if not download:
        raise ValueError("The data directory is not found and download is set to False.")

    # The data is located in a GitHub repository as a zipfile.
    subprocess.run(["git", "clone", URL, path])
    # Remove all git files besides the zipfile
    _remove_other_files(path)

    zip_path = os.path.join(path, "data", "data.zip")
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def get_bitdepth_nucseg_paths(
    path: Union[os.PathLike, str],
    magnification: Optional[Literal['20x', '40x_air', '40x_oil' '63x_oil']] = None,
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the BitDepth NucSeg data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        magnification: The magnification scale for the input images.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_bitdepth_nucseg_data(path, download)

    if magnification is None:
        magnification = "*"
    else:
        if magnification.find("_") != -1:
            _splits = magnification.split("_")
            magnification = f"{_splits[0]} {_splits[1]}"

    raw_paths = natsorted(glob(os.path.join(data_dir, magnification, "images_16bit", "*.tif")))
    label_paths = natsorted(glob(os.path.join(data_dir, magnification, "label masks", "*.tif")))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_bitdepth_nucseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    magnification: Optional[Literal['20x', '40x_air', '40x_oil' '63x_oil']] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BitDepth NucSeg dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        magnification: The magnification scale for the input images.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_bitdepth_nucseg_paths(path, magnification, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_bitdepth_nucseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    magnification: Optional[Literal['20x', '40x_air', '40x_oil' '63x_oil']] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BitDepth NucSeg dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        magnification: The magnification scale for the input images.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bitdepth_nucseg_dataset(path, patch_shape, magnification, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
