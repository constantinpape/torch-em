"""The PARSE dataset contains annotations for pulmonary artery segmentation in
contrast-enhanced CT pulmonary angiography (CTPA) scans.

It comprises the training set of the PARSE2022 challenge (https://parse2022.grand-challenge.org):
100 CTPA volumes with a refined multi-level pulmonary artery annotation. The 30 validation volumes and
70 test volumes of the challenge are distributed without annotations and are therefore not included here.

NOTE: The label legend is as follows:
- background: 0, pulmonary artery: 1

The dataset is located at https://parse2022.grand-challenge.org/Dataset/, from where the organizers share
the training set via a public Google Drive link (mirrored on Baidu Netdisk).

This dataset is from the publication https://doi.org/10.48550/arXiv.2304.03708.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://drive.google.com/uc?id=1_-w8kNc2k4ttHTrVnRWaEaLbjGmNWRZD"
CHECKSUM = "4d51cae38b4e9ca530d9f577dea0e77e3e08d5610bfac0bd632f173a07554468"

LABEL_IDS = {"background": 0, "pulmonary_artery": 1}


def get_parse22_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PARSE dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    data_dir = os.path.join(path, "train")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    rar_path = os.path.join(path, "train.rar")
    util.download_source_gdrive(path=rar_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip_rarfile(rar_path=rar_path, dst=path, remove=True)

    return data_dir


def get_parse22_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the PARSE data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_parse22_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "*", "image", "*.nii.gz")))
    label_paths = [p.replace(os.sep + "image" + os.sep, os.sep + "label" + os.sep) for p in raw_paths]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_parse22_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PARSE dataset for pulmonary artery segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_parse22_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_parse22_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PARSE dataloader for pulmonary artery segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_parse22_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
