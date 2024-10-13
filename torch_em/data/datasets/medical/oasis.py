"""
"""

import os
from glob import glob
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.v1.0.tar"
CHECKSUM = "86dd117dda17f736ade8a4088d7e98e066e1181950fe8b406f1a35f7fb743e78"


def get_oasis_data(path: Union[os.PathLike, str], download: bool = False):
    """
    """
    data_path = os.path.join(path, "data")
    if os.path.exists(data_path):
        return

    os.makedirs(path, exist_ok=True)
    util.download_source(path=path, url=URL, download=download, checksum=CHECKSUM)
    tar_path = os.path.join(path, "neurite-oasis.v1.0.tar")
    util.unzip_tarfile(tar_path=tar_path, dst=data_path, remove=False)


def get_oasis_paths(
    path: Union[os.PathLike, str],
    source: Literal['orig', 'norm'] = "orig",
    label_annotations: Literal['4', '35'] = "4",
    download: bool = False
) -> Tuple[List[int], List[int]]:
    """
    """
    get_oasis_data(path, download)

    patient_dirs = glob(os.path.join(path, "data", "OASIS_*"))
    raw_paths, label_paths = [], []
    for pdir in patient_dirs:
        raw_paths.append(os.path.join(pdir, f"seg{label_annotations}.nii.gz"))
        label_paths.append(os.path.join(pdir, f"{source}.nii.gz"))

    assert len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def get_oasis_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    source: Literal['orig', 'norm'] = "orig",
    label_annotations: Literal['4', '35'] = "4",
    download: bool = False,
    **kwargs
) -> Dataset:
    """
    """
    raw_paths, label_paths = get_oasis_paths(path, source, label_annotations, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_oasis_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    source: Literal['orig', 'norm'] = "orig",
    label_annotations: Literal['4', '35'] = "4",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_oasis_dataset(path, patch_shape, source, label_annotations, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
