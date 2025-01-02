"""The CartoCell dataset contains annotations of cell segmentation in
whole epithelial cysts in high-content screening microscopy images.

The dataset is located at https://data.mendeley.com/datasets/7gbkxgngpm/2.
This dataset is from the publication https://doi.org/10.1016/j.crmeth.2023.100597.
Please cite it if you use this dataset for your research.
"""

import os
import shutil
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Optional, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/7gbkxgngpm-2.zip"
CHECKSUM = "ca3fc289e7b67febfc03cdd55fd791078f7527820c8dbcee0b98d03d993bb6f5"
DNAME = "CartoCell, a high-content pipeline for accurate 3D image analysis, unveils cell morphology patterns in epithelial cysts"  # noqa


def get_cartocell_data(path: Union[os.PathLike, str], download: bool = False):
    """Download the CartoCell dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "cartocell.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)
    shutil.move(src=os.path.join(path, DNAME), dst=data_dir)


def get_cartocell_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "test"]] = None,
    name: Optional[Literal["eggChambers", "embryoids", "MDCK-Normoxia", "MDCK-Hypoxia"]] = None,
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the CartoCell data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', or 'test'.
        name: The name of data subset. Either 'eggChambers', 'embryoids', 'MDCK-Normoxia' or 'MDCK-Hypoxia'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_cartocell_data(path, download)

    if split is None:
        split = ""
    else:
        split = split + "_"

    if name is None:
        name = "*"
    elif name == "MDCK-Hypoxia":
        raise ValueError(f"'{name}' has mismatching shapes for image and corresponding labels.")

    raw_paths = natsorted(glob(os.path.join(path, "data", f"low-resolution_{name}_{split}raw_images", "*")))

    # NOTE: The 'MDCK-Hypoxia' inputs have mismatching input-label shapes (and axes seem interchanged)
    raw_paths = [rpath for rpath in raw_paths if rpath.find("MDCK-Hypoxia") == -1]
    label_paths = [rpath.replace("raw", "label") for rpath in raw_paths]

    assert len(raw_paths) > 0 and len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def get_cartocell_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Optional[Literal["train", "test"]] = None,
    name: Optional[Literal["eggChambers", "embryoids", "MDCK-Normoxia", "MDCK-Hypoxia"]] = None,
    download: bool = False, **kwargs
) -> Dataset:
    """Get the CartoCell dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', or 'test'.
        name: The name of data subset. Either 'eggChambers', 'embryoids', 'MDCK-Normoxia' or 'MDCK-Hypoxia'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cartocell_paths(path, split, name, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_cartocell_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Optional[Literal["train", "test"]] = None,
    name: Optional[Literal["eggChambers", "embryoids", "MDCK-Normoxia", "MDCK-Hypoxia"]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CartoCell dataloader for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', or 'test'.
        name: The name of data subset. Either 'eggChambers', 'embryoids', 'MDCK-Normoxia' or 'MDCK-Hypoxia'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cartocell_dataset(path, patch_shape, split, name, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
