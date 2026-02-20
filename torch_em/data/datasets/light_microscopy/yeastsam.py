"""The YeastSAM dataset contains annotations for budding yeast cell
instance segmentation in DIC (Differential Interference Contrast) microscopy images.

The dataset provides 44 images with corresponding instance segmentation masks.

The dataset is located at https://zenodo.org/records/17204942.
This dataset is from the publication https://doi.org/10.1101/2025.09.17.676679.
Please cite it if you use this dataset in your research.
"""

import os
from typing import Union, Tuple

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/17204942/files/CLB2.zip?download=1"
CHECKSUM = "dc2f32a1ea79e2f65bc28ce79e41681d734b48d312f7fcf43956c4eae41af774"


def get_yeastsam_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the YeastSAM dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the directory with the data.
    """
    data_dir = os.path.join(path, "DIC")
    if os.path.exists(data_dir):
        return path

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "CLB2.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return path


def get_yeastsam_paths(
    path: Union[os.PathLike, str],
    download: bool = False,
) -> Tuple[str, str]:
    """Get paths to the YeastSAM data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where image data is stored.
        Filepath to the folder where label data is stored.
    """
    get_yeastsam_data(path, download)

    image_folder = os.path.join(path, "DIC")
    label_folder = os.path.join(path, "DIC_mask")

    return image_folder, label_folder


def get_yeastsam_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the YeastSAM dataset for yeast cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_folder, label_folder = get_yeastsam_paths(path, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True,
    )
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_folder,
        raw_key="*.tif",
        label_paths=label_folder,
        label_key="*.tif",
        patch_shape=patch_shape,
        is_seg_dataset=False,
        ndim=2,
        **kwargs
    )


def get_yeastsam_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the YeastSAM dataloader for yeast cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_yeastsam_dataset(
        path=path,
        patch_shape=patch_shape,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
