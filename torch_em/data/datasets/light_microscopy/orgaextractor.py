"""
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


# NOTE: The odd thing is, 'val' has no labels, but 'test' has labels.
# So, users are allowed to only request for 'train' and 'test' splits.
URLS = {
    "train": "https://drive.google.com/uc?export=download&id=1u987UNcZxWkEwe5gjLoR3-M0lBNicXQ1",
    "val": "https://drive.google.com/uc?export=download&id=1UsBrHOYY0Orkb4vsRP8SaDj-CeYfGpFG",
    "test": "https://drive.google.com/uc?export=download&id=1IXqu1MqMZzfw1_GzZauUhg1As_abbk6N",
}

CHECKSUMS = {
    "train": None,
    "val": "3d2288a7be39a692af2eb86bea520e7db332191cd372a8c970679b5bede61b7e",
    "test": None,
}


def get_orgaextractor_data(
    path: Union[os.PathLike, str], split: Literal["train", "val"] = None, download: bool = False,
) -> str:
    """
    """
    data_dir = os.path.join(path, split)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(data_dir, f"{split}.zip")
    util.download_source_gdrive(
        path=zip_path, url=URLS[split], download=download, checksum=CHECKSUMS[split], download_type="zip",
    )
    util.unzip(zip_path=zip_path, dst=data_dir, remove=False)
    return data_dir


def get_orgaextractor_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val"] = None, download: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    """
    data_dir = get_orgaextractor_data(path, split, download)

    image_paths = natsorted(glob(os.path.join(data_dir, "*.png")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "*.tif")))

    assert image_paths and len(image_paths) == len(gt_paths)

    return image_paths, gt_paths


def get_orgaextractor_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """
    """
    image_paths, gt_paths = get_orgaextractor_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        ndim=2,
        **kwargs
    )


def get_orgaextractor_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_orgaextractor_dataset(path, patch_shape, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
