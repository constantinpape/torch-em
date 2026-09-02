"""The CAMUS dataset contains annotations for cardiac structures segmentation in 2d echocardiography images.

The database is located at:
https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8.
This dataset is from the publication https://doi.org/10.1109/TMI.2019.2900516.
Please cite it if you use this dataset for a publication.
"""

import os
from glob import glob
from typing import Union, Tuple, Optional, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/folder/63fde55f73e9f004868fb7ac/download"

# TODO: the checksums are different with each download, not sure why
# CHECKSUM = "43745d640db5d979332bda7f00f4746747a2591b46efc8f1966b573ce8d65655"


def get_camus_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Get the CAMUS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "database_nifti")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "CAMUS.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=None)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_camus_paths(
    path: Union[os.PathLike, str], chamber: Optional[Literal[2, 4]] = None, download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the CAMUS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        chamber: The choice of chamber.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_camus_data(path=path, download=download)

    if chamber is None:
        chamber = "*"  # 2CH / 4CH
    else:
        assert chamber in [2, 4], f"{chamber} is  not a valid chamber choice for the acquisitions."
        chamber = f"{chamber}CH"

    image_paths = sorted(glob(os.path.join(data_dir, "patient*", f"patient*_{chamber}_half_sequence.nii.gz")))
    gt_paths = sorted(glob(os.path.join(data_dir, "patient*", f"patient*_{chamber}_half_sequence_gt.nii.gz")))

    return image_paths, gt_paths


def get_camus_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    chamber: Optional[Literal[2, 4]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CAMUS dataset for cardiac structure segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        chamber: The choice of chamber.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_camus_paths(path, chamber, download)

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


def get_camus_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    chamber: Optional[Literal[2, 4]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CAMUS dataloader for cardiac structure segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        chamber: The choice of chamber.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_camus_dataset(path, patch_shape, chamber, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
