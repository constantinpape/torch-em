import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Optional

import torch_em

from .. import util


URL = {
    "train": "https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip",
    "test": "https://zenodo.org/records/3431873/files/CHAOS_Test_Sets.zip"
}

CHECKSUM = {
    "train": "535f7d3417a0e0f0d9133fb3d962423d2a9cf3f103e4f09a3d8a1daf87d5d2fc",
    "test": "80e9e4d4c4e363f142de4570e9b698e3f92dcb5140cc25a9c1cf4963e5ae7541"
}


def get_chaos_data(path, split, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "data", "Train_Sets" if split == "train" else "Test_Sets")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, f"chaos_{split}.zip")
    util.download_source(path=zip_path, url=URL[split], download=download, checksum=CHECKSUM[split])
    util.unzip(zip_path=zip_path, dst=os.path.join(path, "data"))

    return data_dir


def _get_chaos_paths(path, split, modality, download):
    data_dir = get_chaos_data(path=path, split=split, download=download)

    if modality is None:
        modality = ["CT", "MRI"]
    else:
        if isinstance(modality, str):
            modality = [modality]

    for m in modality:
        if m.upper() == "CT":
            ...
        elif m.upper() == "MRI":
            ...
        else:
            raise ValueError

        series_uids = natsorted(glob(os.path.join(data_dir, m.upper(), "*")))

        breakpoint()

    image_paths = ...
    gt_paths = ...

    return image_paths, gt_paths


def get_chaos_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: str,
    modality: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """
    """
    image_paths, gt_paths = _get_chaos_paths(
        path=path, split=split, download=download, modality=modality, download=download
    )

    dataset = ...

    return dataset


def get_chaos_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    split: str,
    modality: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_chaos_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        modality=modality,
        resize_inputs=resize_inputs,
        download=download,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
