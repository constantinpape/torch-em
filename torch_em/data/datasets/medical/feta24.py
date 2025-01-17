"""The FETA24 contains annotations for fetal brain tissue segmentation in MRI.

This dataset is from FeTa 2024 Challenge: https://doi.org/10.5281/zenodo.11192452.
The dataset is from the publication https://doi.org/10.1038/s41597-021-00946-3.
Please cite it if you use this dataset in your publication.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def get_feta24_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Obtain the FETA24 data.
    
    This function describes the download functionality and ensures your data has been downloaded in expected format.

    The dataset is from the FeTa Challenge 2024 - https://fetachallenge.github.io/ (Task 1: Segmentation).
    A detailed description of the dataset is provided here: https://fetachallenge.github.io/pages/Data_description.
    To download the dataset, please follow the below mentioned steps:
    - Go to the section `1. Request access and download the FeTa 2024 data from the University Children's Hospital
    Zurich` at `https://fetachallenge.github.io/pages/Data_download`, which explains the steps to be a registered user
    in Synapse platform and expects the user to agree with the mentioned conditions.
    - While registration, the users are expected to provide some information
    (see https://fetachallenge.github.io/pages/Data_download for details).
    - Next, you can proceed with requesting access (by following provided instructions) at
    https://www.synapse.org/#!Synapse:syn25649159/wiki/610007.

    Once you have access to the dataset, you can use the synapse client or the platform download option to get
    the zipped files. It contains 80 scans paired with their segmentations (more details in the challenge website).

    Finally, you should provide the path to the parent directory where the zipfile is stored.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    if download:
        print("Download is not supported due to the challenge's setup. See 'get_feta24_data' for details.")

    data_dir = os.path.join(path, "feta_2.3")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "feta_2.3.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"The downloaded zip file was not found. Please download it and place it at '{path}'.")

    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_feta24_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get the FETA24 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_feta24_data(path=path, download=download)

    base_dir = os.path.join(data_dir, "sub-*", "anat")
    image_paths = natsorted(glob(os.path.join(base_dir, "sub-*_rec-*_T2w.nii.gz")))
    gt_paths = natsorted(glob(os.path.join(base_dir, "sub-*_rec-*_dseg.nii.gz")))

    return image_paths, gt_paths


def get_feta24_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the FETA24 dataset for fetal brain tissue segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_feta24_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        **kwargs
    )


def get_feta24_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the FETA24 dataloader for fetal brain tissue segmentation.

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
    dataset = get_feta24_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
