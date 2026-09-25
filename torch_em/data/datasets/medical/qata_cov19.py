"""The QaTa-COV19 dataset contains annotations for COVID-19 infection region segmentation in chest X-rays.

The dataset (QaTa-COV19-v1 subset) consists of 4,603 COVID-19 chest X-rays, 2,951 of which have
corresponding ground-truth segmentation masks for the infected lung regions. The masks are binary,
where the foreground marks the COVID-19 infection region.

This dataset is located at https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset.
NOTE: There is another mirror of this dataset on Kaggle (`prashant268/chest-xray-covid19-pneumonia`)
that only provides classification labels, without any segmentation masks. Please make sure to use the
dataset mentioned above for the segmentation task.

This dataset is from the publication https://doi.org/10.1007/s13755-021-00146-8.
Please cite it if you use this dataset for your research.
"""

import os
import zipfile
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


KAGGLE_DATASET_NAME = "aysendegerli/qatacov19-dataset"


def _extract_v1_subset(zip_path, path):
    with zipfile.ZipFile(zip_path) as f:
        members = [m for m in f.namelist() if "/QaTa-COV19-v1/" in m]
        f.extractall(path, members=members)


def get_qata_cov19_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the QaTa-COV19 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "QaTa-COV19", "QaTa-COV19-v1")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "qatacov19-dataset.zip")
    util.download_source_kaggle(path=path, dataset_name=KAGGLE_DATASET_NAME, download=download)
    _extract_v1_subset(zip_path, path)
    os.remove(zip_path)

    if not os.path.exists(data_dir):
        raise RuntimeError(f"The dataset could not be found at '{data_dir}' after extraction.")

    return data_dir


def get_qata_cov19_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the QaTa-COV19 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_qata_cov19_data(path=path, download=download)

    gt_paths = natsorted(glob(os.path.join(data_dir, "Ground-truths", "mask_*.png")))
    image_paths = [
        os.path.join(data_dir, "Images", os.path.basename(p)[len("mask_"):]) for p in gt_paths
    ]

    if len(image_paths) == 0 or len(image_paths) != len(gt_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return image_paths, gt_paths


def get_qata_cov19_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the QaTa-COV19 dataset for COVID-19 infection region segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_qata_cov19_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs
    )


def get_qata_cov19_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the QaTa-COV19 dataloader for COVID-19 infection region segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_qata_cov19_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
