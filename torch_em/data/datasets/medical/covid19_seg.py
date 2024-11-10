"""The Covid19Seg dataset contains annotations for lung and covid infection in CT scans.

This dataset is located at https://doi.org/10.5281/zenodo.3757476.
The dataset is from the publication https://doi.org/10.1002/mp.14676.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, Optional, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = {
    "images": "https://zenodo.org/records/3757476/files/COVID-19-CT-Seg_20cases.zip",
    "lung_and_infection": "https://zenodo.org/records/3757476/files/Lung_and_Infection_Mask.zip",
    "lung": "https://zenodo.org/records/3757476/files/Lung_Mask.zip",
    "infection": "https://zenodo.org/records/3757476/files/Infection_Mask.zip"
}

CHECKSUM = {
    "images": "a5060480eff9315b069b086312dac4872777901fb80d268a5a83edd9f4e7b440",
    "lung_and_infection": "34f5a573cb8fb53cb15abe81868395d9addf436854826a6fd6e70c2b294f19c3",
    "lung": "f060b0d0299939a6d95ddefdbfa281de1a779c4d230a5adbd32414711d6d8187",
    "infection": "87901c73fdd2230260e61d2dbc57bf56026efc28264006b8ea2bf411453c1694"
}

ZIP_FNAMES = {
    "images": "COVID-19-CT-Seg_20cases.zip",
    "lung_and_infection": "Lung_and_Infection_Mask.zip",
    "lung": "Lung_Mask.zip",
    "infection": "Infection_Mask.zip"
}


def get_covid19_seg_data(
    path: Union[os.PathLike, str], task: Literal['lung', 'infection', 'lung_and_infection'], download: bool = False
) -> Tuple[str, str]:
    """Download the Covid19Seg dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        task: The choice of labels for specific task.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the image data is downloaded.
        Filepath where the label data is downloaded.
    """
    im_dir = os.path.join(path, "images", Path(ZIP_FNAMES["images"]).stem)
    gt_dir = os.path.join(path, "gt", Path(ZIP_FNAMES[task]).stem)

    if os.path.exists(im_dir) and os.path.exists(gt_dir):
        return im_dir, gt_dir

    os.makedirs(path, exist_ok=True)

    im_zip_path = os.path.join(path, ZIP_FNAMES["images"])
    gt_zip_path = os.path.join(path, ZIP_FNAMES[task])

    # download the images
    util.download_source(path=im_zip_path, url=URL["images"], download=download, checksum=CHECKSUM["images"])
    util.unzip(zip_path=im_zip_path, dst=im_dir, remove=False)

    # download the labels
    util.download_source(path=gt_zip_path, url=URL[task], download=download, checksum=CHECKSUM[task])
    util.unzip(zip_path=gt_zip_path, dst=gt_dir)

    return im_dir, gt_dir


def get_covid19_seg_paths(
    path: Union[os.PathLike, str], task: Literal['lung', 'infection', 'lung_and_infection'], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Covid19Seg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        task: The choice of labels for specific task.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if task is None:
        task = "lung_and_infection"
    else:
        assert task in ["lung", "infection", "lung_and_infection"], f"{task} is not a valid task."

    image_dir, gt_dir = get_covid19_seg_data(path=path, task=task, download=download)

    image_paths = natsorted(glob(os.path.join(image_dir, "*.nii.gz")))
    gt_paths = natsorted(glob(os.path.join(gt_dir, "*.nii.gz")))

    return image_paths, gt_paths


def get_covid19_seg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    task: Optional[str] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Covid19Seg dataset for lung and covid infection segmentation in CT scans.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        task: The choice of labels for specific task.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_covid19_seg_paths(path, task, download)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )

    return dataset


def get_covid19_seg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    task: Optional[str] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Covid19Seg dataloader for lung and covid infection segmentation in CT scans.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        task: The choice of labels for specific task.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_covid19_seg_dataset(path, patch_shape, task, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
