import os
from glob import glob
from typing import Union, Tuple, Optional

import torch_em
from torch_em.transform.generic import ResizeInputs

from .. import util
from ... import ImageCollectionDataset


URL = "https://scholar.cu.edu.eg/Dataset_BUSI.zip"
CHECKSUM = None


def get_busi_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "Dataset_BUSI_with_GT")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "Dataset_BUSI.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM, verify=False)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_busi_paths(path, category, download):
    data_dir = get_busi_data(path=path, download=download)

    if category is None:
        category = "*"

    data_dir = os.path.join(data_dir, category)

    image_paths = sorted(glob(os.path.join(data_dir, r"*).png")))
    gt_paths = sorted(glob(os.path.join(data_dir, r"*)_mask.png")))

    return image_paths, gt_paths


def get_busi_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    category: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """"Dataset for segmentation of breast cancer in ultrasound images.

    This database is located at https://scholar.cu.edu.eg/?q=afahmy/pages/dataset

    The dataset is from Al-Dhabyani et al. - https://doi.org/10.1016/j.dib.2019.104863
    Please cite it if you use this dataset for a publication.
    """
    if category is not None:
        assert category in ["normal", "benign", "malignant"]

    image_paths, gt_paths = _get_busi_paths(path=path, category=category, download=download)

    if resize_inputs:
        raw_trafo = ResizeInputs(target_shape=patch_shape, is_rgb=True)
        label_trafo = ResizeInputs(target_shape=patch_shape, is_label=True)
        patch_shape = None
    else:
        patch_shape = patch_shape
        raw_trafo, label_trafo = None, None

    dataset = ImageCollectionDataset(
        raw_image_paths=image_paths,
        label_image_paths=gt_paths,
        patch_shape=patch_shape,
        raw_transform=raw_trafo,
        label_transform=label_trafo,
        **kwargs
    )

    return dataset


def get_busi_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    category: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of breast cancer in ultrasound images. See `get_busi_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_busi_dataset(
        path=path,
        patch_shape=patch_shape,
        category=category,
        resize_inputs=resize_inputs,
        download=download,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
