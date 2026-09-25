"""The ATLAS dataset contains annotations for liver and liver tumor segmentation in
contrast-enhanced T1-weighted MRI of patients with hepatocellular carcinoma.

The training set that is released for the ATLAS challenge consists of 60 CE-MRI volumes with manual
delineations of the liver and of the liver tumors. The label ids are: 0 = background, 1 = liver, 2 = tumor.
They are taken from the 'dataset.json' of the official release, which also states that the images are T1w.
NOTE: The two labels are disjoint, i.e. the liver label covers the parenchyma without the tumors, so a mask of
the whole liver is obtained by combining both ids. This was verified on all 60 volumes of the training set.
The test set of 30 volumes is not part of the release.

NOTE: The dataset is located at https://atlas-challenge.u-bourgogne.fr/dataset and is only available to
registered users, so it cannot be downloaded automatically. To download the dataset, please follow these steps:
- Visit https://atlas-challenge.u-bourgogne.fr/dataset and create an account via 'Log in / Sign up'.
- Log in and download the training set from the 'Get the ATLAS dataset' section.
- Place the downloaded zip file (e.g. 'atlas-train-dataset-1.0.1.zip') in the folder passed as 'path'.
  This module unzips it and finds the 'imagesTr' and 'labelsTr' folders in the extracted data.

The data is licensed under CC BY-NC-SA 4.0.

This dataset is from the publication https://doi.org/10.3390/data8050079.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


LABEL_IDS = {"background": 0, "liver": 1, "tumor": 2}


def _find_image_dir(path):
    """Find the 'imagesTr' folder of the manually downloaded data, which is nested in the release folder."""
    image_dirs = glob(os.path.join(path, "**", "imagesTr"), recursive=True)
    return image_dirs[0] if image_dirs else None


def get_atlas_liver_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Obtain the ATLAS liver dataset.

    Args:
        path: Filepath to a folder where the manually downloaded data is stored.
        download: Whether to download the data if it is not present. The data cannot be downloaded
            automatically, so this raises if the data has not been downloaded manually.

    Returns:
        Filepath to the folder with the 'imagesTr' and 'labelsTr' folders.
    """
    image_dir = _find_image_dir(path)
    if image_dir is not None:
        return os.path.dirname(image_dir)

    zip_paths = glob(os.path.join(path, "*atlas*.zip")) + glob(os.path.join(path, "*ATLAS*.zip"))
    if not zip_paths:
        msg = "'torch_em' cannot download this dataset, because the ATLAS data is only available to users "
        msg += "registered at 'https://atlas-challenge.u-bourgogne.fr/dataset'. Please create an account there, "
        msg += f"download the training set and place the zip file (e.g. 'atlas-train-dataset-1.0.1.zip') in '{path}'."
        raise NotImplementedError(msg)

    util.unzip(zip_path=zip_paths[0], dst=path, remove=False)

    image_dir = _find_image_dir(path)
    if image_dir is None:
        raise FileNotFoundError(f"Could not find an 'imagesTr' folder in the data extracted to '{path}'.")

    return os.path.dirname(image_dir)


def get_atlas_liver_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the ATLAS liver data.

    Args:
        path: Filepath to a folder where the manually downloaded data is stored.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_atlas_liver_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    label_paths = natsorted(glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_atlas_liver_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ATLAS liver dataset for liver and liver tumor segmentation.

    Args:
        path: Filepath to a folder where the manually downloaded data is stored.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_atlas_liver_paths(path, download)

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


def get_atlas_liver_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ATLAS liver dataloader for liver and liver tumor segmentation.

    Args:
        path: Filepath to a folder where the manually downloaded data is stored.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_atlas_liver_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
