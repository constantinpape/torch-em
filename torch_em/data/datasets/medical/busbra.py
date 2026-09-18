"""The BUS-BRA dataset contains annotations for breast tumor segmentation in ultrasound images.

The dataset contains 1,875 breast ultrasound images from 1,064 patients, with biopsy-confirmed
tumor region masks (benign or malignant). The images were acquired at the National Institute of
Cancer (INCA, Brazil) with different ultrasound scanners.

This dataset is located at https://zenodo.org/records/8231412 (CC BY 4.0). See also
https://github.com/wgomezf/BUS-BRA for further details on the dataset.

This dataset is from the publication https://doi.org/10.1002/mp.16812.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/8231412/files/BUSBRA.zip?download=1"
CHECKSUM = "ba3e6ed19cc37c682d8d39e25435bbf8a555a12cb7e641b5f2117685c95580ff"


def get_busbra_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BUS-BRA dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "BUSBRA")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "BUSBRA.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_busbra_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the BUS-BRA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_busbra_data(path=path, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "Images", "*.png")))
    gt_paths = sorted(glob(os.path.join(data_dir, "Masks", "*.png")))

    if len(image_paths) == 0 or len(image_paths) != len(gt_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return image_paths, gt_paths


def get_busbra_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BUS-BRA dataset for breast tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_busbra_paths(path, download)

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


def get_busbra_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BUS-BRA dataloader for breast tumor segmentation.

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
    dataset = get_busbra_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
