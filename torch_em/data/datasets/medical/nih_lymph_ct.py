import os
from glob import glob
from typing import Tuple, Union

import torch_em

from .. import util


def get_nih_lymph_ct_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "")
    # if os.path.exists(data_dir):
    #     return data_dir

    # tcia download
    # unzip files, if required

    return data_dir


def _get_nih_lymph_ct_paths(path, download):
    data_dir = get_nih_lymph_ct_data(path=path, download=download)

    image_paths = ...
    gt_paths = ...

    return image_paths, gt_paths


def get_nih_lymph_ct_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    download: bool = False,
    **kwargs
):
    image_paths, gt_paths = _get_nih_lymph_ct_paths(path=path, download=download)

    dataset = ...

    return dataset


def get_nih_lymph_ct_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    download: bool = False,
    **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_nih_lymph_ct_dataset(path=path, patch_shape=patch_shape, download=download, **ds_kwargs)
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
