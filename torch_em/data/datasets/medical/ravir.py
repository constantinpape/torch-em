import os
import shutil
from glob import glob
from typing import Union, Tuple

import torch_em

from .. import util


URL = "https://drive.google.com/uc?export=download&id=1ZlZoSStvE9VCRq3bJiGhQH931EF0h3hh"
CHECKSUM = "b9cc2e84660ab4ebeb583d510bd71057faf596a99ed6d1e27aee361e3a3f1381"


def get_ravir_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "RAVIR_Dataset")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "ravir.zip")
    util.download_source_gdrive(
        path=zip_path, url=URL, download=download, checksum=CHECKSUM, download_type="zip"
    )
    util.unzip(zip_path=zip_path, dst=path)

    # updating the folder structure a tiny bit
    tmp_dir = os.path.join(path, r"RAVIR Dataset")
    assert os.path.exists(tmp_dir), "Something went wrong with the data download"
    shutil.move(tmp_dir, data_dir)

    return data_dir


def _get_ravir_paths(path, download):
    data_dir = get_ravir_data(path=path, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "train", "training_images", "*")))
    gt_paths = sorted(glob(os.path.join(data_dir, "train", "training_masks", "*")))

    return image_paths, gt_paths


def get_ravir_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    download: bool = False,
    resize_inputs: bool = False,
    **kwargs
):
    """Dataset for segmentation of retinal arteries and veins in infrared reflectance (IR) imaging.

    This dataset comes from the "RAVIR" challenge:
    - https://ravir.grand-challenge.org/RAVIR/
    - https://doi.org/10.1109/JBHI.2022.3163352

    Please cite it if you use this dataset for a publication.
    """
    image_paths, gt_paths = _get_ravir_paths(path=path, download=download)

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


def get_ravir_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    download: bool = False,
    resize_inputs: bool = False,
    **kwargs
):
    """Dataloader for segmentation of retinal arteries and veins in IR imaging. See `get_ravir_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ravir_dataset(
        path=path, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
