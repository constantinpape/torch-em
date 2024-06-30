import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple

import torch_em

from .. import util


URL = "https://zenodo.org/records/10475293/files/Micro_Ultrasound_Prostate_Segmentation_Dataset.zip?"
CHECKSUM = "031645dc30948314e379d0a0a7d54bad1cd4e1f3f918b77455d69810aa05dce3"


def get_micro_usp_data(path, download):
    os.makedirs(path, exist_ok=True)

    fname = Path(URL).stem
    data_dir = os.path.join(path, fname)
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, f"{fname}.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_micro_usp_paths(path, split, download):
    data_dir = get_micro_usp_data(path=path, download=download)

    image_paths = natsorted(glob(os.path.join(data_dir, split, "micro_ultrasound_scans", "*.nii.gz")))
    gt_paths = natsorted(glob(os.path.join(data_dir, split, "expert_annotations", "*.nii.gz")))

    return image_paths, gt_paths


def get_micro_usp_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: str,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of prostate in micro-ultrasound scans.

    This dataset is from Jiang et al. - https://doi.org/10.1016/j.compmedimag.2024.102326.
    Please cite it if you use this dataset for a publication.
    """
    image_paths, gt_paths = _get_micro_usp_paths(path=path, split=split, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_micro_usp_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    split: str,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of prostate in micro-ultrasound scans. See `get_micro_usp_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_micro_usp_dataset(
        path=path, patch_shape=patch_shape, split=split, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
