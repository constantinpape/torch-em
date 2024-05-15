import os
from glob import glob
from typing import Union, Tuple, Optional

import torch_em
from torch_em.transform.generic import ResizeInputs

from .. import util


URL = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/folder/63fde55f73e9f004868fb7ac/download"
CHECKSUM = "43745d640db5d979332bda7f00f4746747a2591b46efc8f1966b573ce8d65655"


def get_camus_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "database_nifti")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "CAMUS.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_camus_paths(path, chamber, download):
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
    chamber: Optional[int] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """
    """
    image_paths, gt_paths = _get_camus_paths(path=path, chamber=chamber, download=download)

    if resize_inputs:
        raw_trafo = ResizeInputs(target_shape=patch_shape, is_label=False)
        label_trafo = ResizeInputs(target_shape=patch_shape, is_label=True)
        patch_shape = None
    else:
        patch_shape = patch_shape
        raw_trafo, label_trafo = None, None

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        raw_transform=raw_trafo,
        label_transform=label_trafo,
        **kwargs
    )

    return dataset


def get_camus_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    chamber: Optional[int] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_camus_dataset(
        path=path, patch_shape=patch_shape, chamber=chamber, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
