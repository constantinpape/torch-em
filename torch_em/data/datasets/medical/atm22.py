"""The ATM dataset contains annotations for airway tree segmentation in chest CT scans.

It comprises the training set of the ATM'22 challenge (https://atm22.grand-challenge.org): 300 chest CT
volumes with a full airway tree annotation, distributed in two batches on Zenodo under the CC BY 4.0 license.
The scans are collected from multiple sites and a part of them stems from LIDC-IDRI and EXACT'09.
The images of the 20 EXACT'09 cases (ATM_242 - ATM_250 and ATM_501 - ATM_511) are not redistributed by the
organizers, so only the 280 cases with both an image and an annotation are exposed by this dataset.

NOTE: The label legend is as follows:
- background: 0, airway: 1

The dataset is located at https://doi.org/10.5281/zenodo.7949582 (TrainBatch1) and
https://doi.org/10.5281/zenodo.7949571 (TrainBatch2). See https://atm22.grand-challenge.org for the challenge.

This dataset is from the publication https://doi.org/10.1016/j.media.2023.102957.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "TrainBatch1": "https://zenodo.org/api/records/7949582/files/TrainBatch1.rar/content",
    "TrainBatch2": "https://zenodo.org/api/records/7949571/files/TrainBatch2.rar/content",
}

CHECKSUMS = {
    "TrainBatch1": "355f8c5b19f2b481a2294f0b7c06107893bba198b1ac346d033085eb1d705a20",
    "TrainBatch2": "83a2fee661f80c29783811e2dc7913b45aad38ed19f3341c1686a6609fe815e5",
}

LABEL_IDS = {"background": 0, "airway": 1}


def get_atm22_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ATM dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    os.makedirs(path, exist_ok=True)

    for batch, url in URLS.items():
        if os.path.exists(os.path.join(path, batch)):
            continue

        rar_path = os.path.join(path, f"{batch}.rar")
        util.download_source(path=rar_path, url=url, download=download, checksum=CHECKSUMS[batch])
        util.unzip_rarfile(rar_path=rar_path, dst=path, remove=True)

    return path


def get_atm22_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the ATM data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_atm22_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "TrainBatch*", "imagesTr", "*.nii.gz")))
    label_paths = [p.replace(os.sep + "imagesTr" + os.sep, os.sep + "labelsTr" + os.sep) for p in raw_paths]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_atm22_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ATM dataset for airway tree segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_atm22_paths(path, download)

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


def get_atm22_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ATM dataloader for airway tree segmentation.

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
    dataset = get_atm22_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
