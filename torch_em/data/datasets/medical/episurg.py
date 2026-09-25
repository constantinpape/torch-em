"""The EPISURG dataset contains annotations for resection cavity segmentation in postoperative
brain MRI of epilepsy patients.

The dataset consists of 430 postoperative T1-weighted MRI from patients who underwent resective
brain surgery for refractory epilepsy at the National Hospital of Neurology and Neurosurgery
(Queen Square, London, United Kingdom). The corresponding preoperative MRI is present for 269 of
these subjects. The resection cavity was manually segmented by three human raters on partially
overlapping subsets of the postoperative scans (133, 34 and 33 subjects, respectively), so that
200 of the 430 subjects have a resection cavity mask.

The dataset is located at https://doi.org/10.5522/04/9996158.v1 and is distributed under the
CC BY-NC-SA 4.0 license.
The dataset is from the publication https://doi.org/10.1007/s11548-021-02420-2.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://s3-eu-west-1.amazonaws.com/pstorage-ucl-2748466690/26153588/EPISURG.zip"
CHECKSUM = "91c6e0698ab5a1874662e3a53ccfb509e41718ddb6ac62f0b35f06a85d1c8daf"


def get_episurg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the EPISURG dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "EPISURG")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "EPISURG.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_episurg_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the EPISURG data.

    Only the postoperative subjects with a resection cavity mask from one of the three human
    raters are returned.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the postoperative MRI data.
        List of filepaths for the resection cavity segmentation data.
    """
    data_dir = get_episurg_data(path, download)

    image_paths, label_paths = [], []
    for label_path in natsorted(glob(os.path.join(data_dir, "subjects", "*", "postop", "*postop-seg-*.nii.gz"))):
        subject_dir = os.path.dirname(label_path)
        image_path = natsorted(glob(os.path.join(subject_dir, "*postop-t1mri-*.nii.gz")))
        assert len(image_path) == 1, f"Could not find a unique postop MRI for '{label_path}'."
        image_paths.append(image_path[0])
        label_paths.append(label_path)

    return image_paths, label_paths


def get_episurg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the EPISURG dataset for resection cavity segmentation in postoperative brain MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_episurg_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_episurg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the EPISURG dataloader for resection cavity segmentation in postoperative brain MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_episurg_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
