import os
from glob import glob
from typing import Tuple
from shutil import move, rmtree

import torch_em

from . import util
from .. import ImageCollectionDataset


URL = "https://zenodo.org/records/10278229/files/OrganoidBasic_v20211206.zip"
CHECKSUM = "d067124d734108e46e18f65daaf17c89cb0a40bdacc6f6031815a6839e472798"


def _download_dataset(path, download):
    zip_path = os.path.join(path, "OrganoidBasic_v20211206.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path, remove=True)

    move(os.path.join(path, "OrganoidBasic_v20211206", "train"), os.path.join(path, "train"))
    move(os.path.join(path, "OrganoidBasic_v20211206", "val"), os.path.join(path, "val"))
    move(os.path.join(path, "OrganoidBasic_v20211206", "eval"), os.path.join(path, "eval"))
    rmtree(os.path.join(path, "OrganoidBasic_v20211206"))


def _get_data_paths(path, split, download):
    split_dir = os.path.join(path, split)

    if not os.path.exists(split_dir):
        _download_dataset(path, download)

    image_paths = sorted(glob(os.path.join(split_dir, "*_img.jpg")))
    label_paths = sorted(glob(os.path.join(split_dir, "*_masks_organoid.png")))

    return image_paths, label_paths


def get_orgasegment_dataset(
    path: str,
    split: str,
    patch_shape: Tuple[int, int],
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for the segmentation of human intestinal organoids.

    This dataset is from the publication https://doi.org/10.1038/s42003-024-05966-4.
    Please cite it if you use this dataset for a publication.
    """
    assert split in ["train", "val", "eval"]

    image_paths, label_paths = _get_data_paths(path, split, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries,
    )
    return ImageCollectionDataset(
        raw_image_paths=image_paths, label_image_paths=label_paths, patch_shape=patch_shape, **kwargs
    )


def get_orgasegment_loader(
    path: str,
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for the segmentation of human intestinal organoids in bright-field images.
    See `get_orgasegment_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_orgasegment_dataset(
        path, split, patch_shape, boundaries, binary, download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
