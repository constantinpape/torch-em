import os
from glob import glob
from typing import Union, Tuple, Literal

import torch_em

from .. import util


KAGGLE_DATASET_NAME = "vbookshelf/pneumothorax-chest-xray-images-and-masks"
CHECKSUM = "1ade68d31adb996c531bb686fb9d02fe11876ddf6f25594ab725e18c69d81538"


def get_siim_acr_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "siim-acr-pneumothorax")
    if os.path.exists(data_dir):
        return data_dir

    util.download_source_kaggle(path=path, dataset_name=KAGGLE_DATASET_NAME, download=download)

    zip_path = os.path.join(path, "pneumothorax-chest-xray-images-and-masks.zip")
    util._check_checksum(path=zip_path, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_siim_acr_paths(path, split, download):
    data_dir = get_siim_acr_data(path=path, download=download)

    assert split in ["train", "test"], f"'{split}' is not a valid split."

    image_paths = sorted(glob(os.path.join(data_dir, "png_images", f"*_{split}_*.png")))
    gt_paths = sorted(glob(os.path.join(data_dir, "png_masks", f"*_{split}_*.png")))

    return image_paths, gt_paths


def get_siim_acr_dataset(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    patch_shape: Tuple[int, int],
    download: bool = False,
    resize_inputs: bool = False,
    **kwargs
):
    """Dataset for pneumothorax segmentation in CXR.

    The database is located at https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks/data

    This dataset is from the "SIIM-ACR Pneumothorax Segmentation" competition:
    https://kaggle.com/competitions/siim-acr-pneumothorax-segmentation

    Please cite it if you use this dataset for a publication.
    """
    image_paths, gt_paths = _get_siim_acr_paths(path=path, split=split, download=download)

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
    dataset.max_sampling_attempts = 5000

    return dataset


def get_siim_acr_loader(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    patch_shape: Tuple[int, int],
    batch_size: int,
    download: bool = False,
    resize_inputs: bool = False,
    **kwargs
):
    """Dataloader for pneumothorax segmentation in CXR. See `get_siim_acr_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_siim_acr_dataset(
        path=path, split=split, patch_shape=patch_shape, download=download, resize_inputs=resize_inputs, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
