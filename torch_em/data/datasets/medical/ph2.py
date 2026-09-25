"""The PH2 dataset contains annotations for skin lesion segmentation in dermoscopic images.

The dataset consists of 200 dermoscopic images acquired at the Dermatology Service of Hospital
Pedro Hispano, Matosinhos, Portugal, together with binary lesion segmentation masks and a
classification of dermoscopic criteria (e.g. common nevus, atypical nevus, melanoma).

The dataset is officially hosted at https://fc.up.pt/addi/ph2%20database.html, which requires
filling out a registration form to obtain the download link. We instead use the mirror at
https://www.kaggle.com/datasets/spacesurfer/ph2-dataset, which preserves the same folder layout
as the original 'PH2Dataset.rar' archive.

This dataset is from the publication https://doi.org/10.1109/EMBC.2013.6610779.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


KAGGLE_DATASET_NAME = "spacesurfer/ph2-dataset"


def get_ph2_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PH2 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "PH2Dataset", "PH2 Dataset images")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name=KAGGLE_DATASET_NAME, download=download)

    zip_path = os.path.join(path, "ph2-dataset.zip")
    util.unzip(zip_path=zip_path, dst=path)

    if not os.path.exists(data_dir):
        raise RuntimeError(f"The dataset could not be found at '{data_dir}' after extraction.")

    return data_dir


def get_ph2_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the PH2 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_ph2_data(path=path, download=download)

    image_paths = natsorted(glob(os.path.join(data_dir, "*", "*_Dermoscopic_Image", "*.bmp")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "*", "*_lesion", "*_lesion.bmp")))

    if len(image_paths) == 0 or len(image_paths) != len(gt_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return image_paths, gt_paths


def get_ph2_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PH2 dataset for skin lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_ph2_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_ph2_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PH2 dataloader for skin lesion segmentation.

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
    dataset = get_ph2_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
