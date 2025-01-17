"""The BUSI dataset contains annotations for breast cancer segmentation in ultrasound images.

This dataset is located at https://scholar.cu.edu.eg/?q=afahmy/pages/dataset.
The dataset is from the publication https://doi.org/10.1016/j.dib.2019.104863.
Please cite it if you use this dataset for a publication.
"""

import os
from glob import glob
from typing import Union, Tuple, Optional, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://scholar.cu.edu.eg/Dataset_BUSI.zip"
CHECKSUM = "b2ce09f6063a31a73f628b6a6ee1245187cbaec225e93e563735691d68654de7"


def get_busi_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BUSI dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Dataset_BUSI_with_GT")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "Dataset_BUSI.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM, verify=False)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_busi_paths(
    path: Union[os.PathLike, str],
    category: Optional[Literal["normal", "benign", "malignant"]] = None,
    download: bool = False
) -> Tuple[List[str, List[str]]]:
    """Get paths to the BUSI data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        category: The choice of data sub-category.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = get_busi_data(path=path, download=download)

    if category is None:
        category = "*"
    else:
        if category not in ["normal", "benign", "malignant"]:
            raise ValueError(f"'{category}' is not a valid category choice.")

    data_dir = os.path.join(data_dir, category)

    image_paths = sorted(glob(os.path.join(data_dir, r"*).png")))
    gt_paths = sorted(glob(os.path.join(data_dir, r"*)_mask.png")))

    return image_paths, gt_paths


def get_busi_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    category: Optional[Literal["normal", "benign", "malignant"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BUSI dataset for breast cancer segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        category: The choice of data sub-category.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_busi_paths(path, category, download)

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


def get_busi_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    category: Optional[Literal["normal", "benign", "malignant"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BUSI dataloader for breast cancer segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        category: The choice of data sub-category.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_busi_dataset(path, patch_shape, category, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
