import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple

import torch_em

from .. import util
from .neurips_cell_seg import to_rgb
from ... import ImageCollectionDataset


URL = "https://www.cellpose.org/dataset"


def _get_cellpose_paths(path, split, choice):
    if choice == "cyto":
        assert split in ["train", "test"], f"'{split}' is not a valid split in '{choice}'."
    elif choice == "cyto2":
        assert split == "train", f"'{split}' is not a valid split in '{choice}'."
    else:
        raise ValueError(f"'{choice}' is not a valid dataset choice.")

    image_paths = natsorted(glob(os.path.join(path, choice, split, "*_img.png")))
    gt_paths = natsorted(glob(os.path.join(path, choice, split, "*_masks.png")))

    return image_paths, gt_paths


def get_cellpose_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    choice: str = "cyto",
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of cells in ...

    Automatic download not possible
    """
    assert choice in ["cyto", "cyto2"]
    assert split in ["train", "test"]
    if download:
        assert NotImplementedError(
            "The dataset cannot be automatically downloaded. ",
            "Please see 'get_cellpose_dataset' in 'torch_em/data/datasets/cellpose.py' for details."
        )

    image_paths, gt_paths = _get_cellpose_paths(path=path, split=split, choice=choice)

    if "raw_transform" not in kwargs:
        raw_transform = torch_em.transform.get_raw_transform(augmentation2=to_rgb)

    if "transform" not in kwargs:
        transform = torch_em.transform.get_augmentations(ndim=2)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        raw_transform=raw_transform,
        transform=transform,
        **kwargs
    )
    dataset = ImageCollectionDataset(
        raw_image_paths=image_paths,
        label_image_paths=gt_paths,
        patch_shape=patch_shape,
        raw_transform=raw_transform,
        transform=transform,
    )

    return dataset


def get_cellpose_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    choice: str = "cyto",
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of cells in ... See `get_cellpose_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cellpose_dataset(
        path=path,
        split=split,
        patch_shape=patch_shape,
        choice=choice,
        download=download,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
