"""The DSAD dataset contains annotations for abdominal organs in laparoscopy images.

This dataset is located at https://springernature.figshare.com/articles/dataset/The_Dresden_Surgical_Anatomy_Dataset_for_abdominal_organ_segmentation_in_surgical_data_science/21702600  # noqa
The dataset is from the publication https://doi.org/10.1038/s41597-022-01719-2.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://springernature.figshare.com/ndownloader/files/38494425"
CHECKSUM = "b8a8ade37d106fc1641a901d1c843806f2d27f9f8e18f4614b043e7e2ca2e40f"


def get_dsad_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """
    """
    data_dir = os.path.join(path, "")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "data.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_dsad_paths(
    path: Union[os.PathLike, str], split: Optional[Literal["train", "val", "test"]] = None, download: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    """
    data_dir = get_dsad_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "*")))
    label_paths = natsorted(glob(os.path.join(data_dir, "*")))

    return raw_paths, label_paths


def get_dsad_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "val", "test"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """
    """
    raw_paths, label_paths = get_dsad_paths(path, split, download)

    ...

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_dsad_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "val", "test"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_dsad_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
