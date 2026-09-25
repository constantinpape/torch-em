"""The MosMedData+ dataset contains annotations for COVID-19 lung lesion segmentation in chest CT scans.

This dataset is a subset of the MosMedData collection (https://mosmed.ai/datasets/covid19_1110/):
50 out of the 1110 chest CT studies come with expert-annotated binary pixel masks for ground-glass
opacifications and consolidations (label ids: background 0, lesion 1). The official release requires
registering with an email address, so this loader relies on the public Kaggle mirror at
https://www.kaggle.com/datasets/mathurinache/mosmeddata-chest-ct-scans-with-covid19, which redistributes
the same volumes and masks as uncompressed NIfTI files (*.nii).

The dataset is from the publication https://doi.org/10.1101/2020.05.20.20100362.
Please cite it if you use this dataset in your research.

The dataset is distributed under the CC BY-NC-ND 3.0 license.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


KAGGLE_DATASET_NAME = "mathurinache/mosmeddata-chest-ct-scans-with-covid19"


def get_mosmed_plus_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the MosMedData+ dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name=KAGGLE_DATASET_NAME, download=download)
    zip_path = os.path.join(path, "mosmeddata-chest-ct-scans-with-covid19.zip")
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def get_mosmed_plus_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the MosMedData+ data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_mosmed_plus_data(path, download)

    mask_paths = natsorted(glob(os.path.join(data_dir, "**", "masks", "study_*_mask.nii"), recursive=True))
    assert len(mask_paths) == 50, f"Expected 50 annotated studies, found {len(mask_paths)} in '{data_dir}'."

    image_paths = []
    for mask_path in mask_paths:
        fname = os.path.basename(mask_path).replace("_mask.nii", ".nii")
        matches = glob(os.path.join(data_dir, "**", "studies", "*", fname), recursive=True)
        assert len(matches) == 1, f"Could not find a unique image volume for '{mask_path}'."
        image_paths.append(matches[0])

    return image_paths, mask_paths


def get_mosmed_plus_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MosMedData+ dataset for COVID-19 lung lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_mosmed_plus_paths(path, download)

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


def get_mosmed_plus_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MosMedData+ dataloader for COVID-19 lung lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mosmed_plus_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
