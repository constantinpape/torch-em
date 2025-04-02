"""The COVID QU EX dataset contains annotations for segmentations of
lung and infection in X-Ray images.

The dataset is located at https://www.kaggle.com/datasets/anasmohammedtahir/covidqu.
This dataset is from the publication https://doi.org/10.1016/j.compbiomed.2021.104319.
Please cite them if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Tuple, Union, Optional, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def get_covid_qu_ex_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the COVID QU EX dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downlaoded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name="anasmohammedtahir/covidqu", download=download)
    zip_path = os.path.join(path, "covidqu.zip")
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def get_covid_qu_ex_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'val', 'test'],
    task: Literal['lung', 'infection'],
    patient_type: Optional[Literal['covid19', 'non-covid', 'normal']] = None,
    segmentation_mask: Literal['lung', 'infection'] = "lung",
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the COVID QU EX data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'train', 'val' or 'test'.
        task: The choice for the subset of dataset. Either 'lung' or 'infection'.
        patient_type: The choice of subset of patients. Either 'covid19', 'non-covid' or 'normal'.
            By default is None, i.e. all the patient data will be chosen.
        segmentation_mask: The choice of segmentation labels. Either 'lung' or 'infection'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_covid_qu_ex_data(path=path, download=download)

    assert split.lower() in ["train", "val", "test"], f"'{split}' is not a valid split."

    if task == "lung":
        _task = r"Lung Segmentation Data/Lung Segmentation Data"
    elif task == "infection":
        _task = r"Infection Segmentation Data/Infection Segmentation Data"
    else:
        raise ValueError(f"'{task}' is not a valid task.")

    if patient_type == "covid19":
        patient_type = "COVID-19"
    elif patient_type == "non-covid":
        patient_type = "Non-COVID"
    elif patient_type == "normal":
        patient_type = "Normal"
    else:
        if patient_type is None:
            patient_type = "*"
        else:
            raise ValueError(f"'{patient_type}' is not a valid patient type.")

    base_dir = os.path.join(data_dir, _task, split.title(), patient_type)

    if segmentation_mask == "lung":
        segmentation_mask = r"lung masks"
    elif segmentation_mask == "infection":
        if task == "lung":
            raise AssertionError("The 'lung' data subset does not have infection masks.")
        segmentation_mask = r"infection masks"
    else:
        if segmentation_mask is None:
            segmentation_mask = "*"
        else:
            raise ValueError(f"'{segmentation_mask}' is not a valid segmentation task.")

    image_paths = natsorted(glob(os.path.join(base_dir, "images", "*")))
    gt_paths = natsorted(glob(os.path.join(base_dir, segmentation_mask, "*")))

    return image_paths, gt_paths


def get_covid_qu_ex_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    task: Literal['lung', 'infection'],
    patient_type: Optional[Literal['covid19', 'non-covid', 'normal']] = None,
    segmentation_mask: Literal['lung', 'infection'] = "lung",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the COVID QU EX dataset for lung and infection segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        task: The choice for the subset of dataset. Either 'lung' or 'infection'.
        patient_type: The choice of subset of patients. Either 'covid19', 'non-covid' or 'normal'.
            By default is None, i.e. all the patient data will be chosen.
        segmentation_mask: The choice of segmentation labels. Either 'lung' or 'infection'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_covid_qu_ex_paths(path, split, task, patient_type, segmentation_mask, download)

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


def get_covid_qu_ex_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    task: Literal['lung', 'infection'],
    patient_type: Optional[Literal['covid19', 'non-covid', 'normal']] = None,
    segmentation_mask: Literal['lung', 'infection'] = "lung",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the COVID QU EX dataloader for lung and infection segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        task: The choice for the subset of dataset. Either 'lung' or 'infection'.
        patient_type: The choice of subset of patients. Either 'covid19', 'non-covid' or 'normal'.
            By default is None, i.e. all the patient data will be chosen.
        segmentation_mask: The choice of segmentation labels. Either 'lung' or 'infection'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_covid_qu_ex_dataset(
        path, patch_shape, split, task, patient_type, segmentation_mask, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
