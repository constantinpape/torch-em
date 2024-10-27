"""
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Sequence

import torch_em

from torch.utils.data import Dataset, DataLoader

from .. import util


URL = {
    "labelled": "https://zenodo.org/record/5153251/files/labelled.zip",
    "unlabelled": "https://zenodo.org/record/5153251/files/unlabelled.zip"
}

CHECKSUMS = {
    "labelled": "e4b6fc8ad3955c4e0fe0e95a9be03d4333b6d9029f675ae9652084cefc4aaab6",
    "unlabelled": "c0228c56140d16141a5f9fb303080861624d6d2d25fab5bd463e489dab9adf4b"
}

VALID_CELL_TYPES = ["A2058", "G361", "HOB", "PC3", "PNT1A"]


def get_vicar_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """
    """
    data_dir = os.path.join(path, "labelled")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(data_dir, exist_ok=True)

    util.download_source(path=path, url=URL["labelled"], download=download, checksum=CHECKSUMS["labelled"])
    util.unzip(zip_path=os.path.join(path, "labelled.zip"), dst=data_dir)

    return data_dir


def get_vicar_paths(
    path: Union[os.PathLike, str],
    cell_types: Optional[Union[Sequence[str], str]] = None,
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """
    """
    data_dir = get_vicar_data(path, download)

    if cell_types is not None and isinstance(cell_types, str):
        raise ValueError("The choice of cell types should be a sequence of string values.")

    if cell_types is None:
        cell_types = VALID_CELL_TYPES
    else:
        if isinstance(cell_types, str):
            cell_types = [cell_types]

    raw_paths, label_paths = [], []
    for cell_type in cell_types:
        assert cell_type in VALID_CELL_TYPES

        raw_paths.extend(natsorted(glob(os.path.join(data_dir, cell_type, "*_img.tif"))))
        label_paths.extend(natsorted(glob(os.path.join(data_dir, cell_type, "*_mask.png"))))

    return raw_paths, label_paths


def get_vicar_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    cell_types: Optional[Union[Sequence[str], str]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """
    """
    raw_paths, label_paths = get_vicar_paths(path, cell_types, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_vicar_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    cell_types: Optional[Union[Sequence[str], str]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_vicar_dataset(path, patch_shape, cell_types, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
