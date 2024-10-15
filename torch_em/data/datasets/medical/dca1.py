import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal

import torch_em

from .. import util


URL = "http://personal.cimat.mx:8181/~ivan.cruz/DB_Angiograms_files/DB_Angiograms_134.zip"
CHECKSUM = "7161638a6e92c6a6e47a747db039292c8a1a6bad809aac0d1fd16a10a6f22a11"


def get_dca1_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "Database_134_Angiograms")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "DB_Angiograms_134.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)
    return data_dir


def _get_dca1_paths(path, split, download):
    data_dir = get_dca1_data(path=path, download=download)

    image_paths, gt_paths = [], []
    for image_path in natsorted(glob(os.path.join(data_dir, "*.pgm"))):
        if image_path.endswith("_gt.pgm"):
            gt_paths.append(image_path)
        else:
            image_paths.append(image_path)

    image_paths, gt_paths = natsorted(image_paths), natsorted(gt_paths)

    if split == "train":  # first 85 images
        image_paths, gt_paths = image_paths[:-49], gt_paths[:-49]
    elif split == "val":  # 15 images
        image_paths, gt_paths = image_paths[-49:-34], gt_paths[-49:-34]
    elif split == "test":  # last 34 images
        image_paths, gt_paths = image_paths[-34:], gt_paths[-34:]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    return image_paths, gt_paths


def get_dca1_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of coronary ateries in x-ray angiography.

    This dataset is from Cervantes-Sanchez et al. - https://doi.org/10.3390/app9245507.

    The database is located at http://personal.cimat.mx:8181/~ivan.cruz/DB_Angiograms.html.

    Please cite it if you use this dataset in a publication.
    """
    image_paths, gt_paths = _get_dca1_paths(path=path, split=split, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )

    return dataset


def get_dca1_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of coronary ateries in x-ray angiography. See `get_dca1_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_dca1_dataset(
        path=path, patch_shape=patch_shape, split=split, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
