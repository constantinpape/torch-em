import os
from glob import glob
from typing import Union, Tuple

import torch_em
from torch_em.transform.generic import ResizeInputs

from .. import util
from ... import ImageCollectionDataset


URL = "https://figshare.com/ndownloader/files/35013982"
CHECKSUM = "15b053dff496bc8e53eb8a8d0707ef73ba3d56c988eea92b65832c9c82852a7d"


def get_papila_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "papila.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_papila_paths(path, download):
    data_dir = get_papila_data(path=path, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "FundusImages", "*.jpg")))
    gt_paths = sorted(glob(os.path.join(data_dir, "ExpertsSegmentations", "ImagesWithContours", "*.jpg")))

    return image_paths, gt_paths


def get_papila_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of optic cup and optic disc in fundus images.

    The database is located at https://figshare.com/articles/dataset/PAPILA/14798004/2

    The dataset is from Kovalyk et al. - https://doi.org/10.1038/s41597-022-01388-1.
    Please cite it if you use this dataset for a publication.
    """
    image_paths, gt_paths = _get_papila_paths(path=path, download=download)

    if resize_inputs:
        raw_trafo = ResizeInputs(target_shape=patch_shape, is_label=False)
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


def get_papila_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of optic cup and optic disc in fundus images. See `get_papila_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_papila_dataset(
        path=path, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
