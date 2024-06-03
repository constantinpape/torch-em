import os
from glob import glob
from natsort import natsorted
from typing import Tuple, Union

import torch_em

from .. import util


URL = "https://zenodo.org/records/11005384/files/acouslic-ai-train-set.zip"
CHECKSUM = "187602dd243a3a872502b57b8ea56e28c67a9ded547b6e816b00c6d41f8b8767"


def get_acouslic_ai_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "acouslic-ai-train-set.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir, remove=False)

    return data_dir


def _get_acouslic_ai_paths(path, download):
    data_dir = get_acouslic_ai_data(path=path, download=download)

    image_paths = natsorted(glob(os.path.join(data_dir, "images", "stacked_fetal_ultrasound", "*.mha")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "masks", "stacked_fetal_abdomen", "*.mha")))

    return image_paths, gt_paths


def get_acouslic_ai_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of fetal abdominal circumference in ultrasound images.

    This dataset is from the ACOUSLIC-AI Challenge: https://acouslic-ai.grand-challenge.org/

    Please cite it if you this dataset for your publication.
    """
    image_paths, gt_paths = _get_acouslic_ai_paths(path=path, download=download)

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
        **kwargs
    )

    return dataset


def get_acouslic_ai_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of fetal abdominal circumference in ultrasound images.
    See `get_acouslic_ai_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_acouslic_ai_dataset(
        path=path, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
