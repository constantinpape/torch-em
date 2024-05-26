import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple

import torch_em

from .. import util


def get_m2caiseg_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, r"m2caiSeg dataset")
    if os.path.exists(data_dir):
        return data_dir

    util.download_source_kaggle(path=path, dataset_name="salmanmaq/m2caiseg", download=download)
    zip_path = os.path.join(path, "m2caiseg.zip")
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _get_m2caiseg_paths(path, split, download):
    data_dir = get_m2caiseg_data(path=path, download=download)

    if split == "val":
        split = "trainval"

    image_paths = natsorted(glob(os.path.join(data_dir, split, "images", "*.jpg")))
    gt_paths = natsorted(glob(os.path.join(data_dir, split, "groundtruth", "*.png")))

    import imageio.v3 as imageio

    for gt_path in gt_paths:
        image = imageio.imread(image_paths[0])
        gt = imageio.imread(gt_path)

        import napari
        v = napari.Viewer()
        v.add_image(image.transpose(2, 0, 1))
        v.add_labels(gt.transpose(2, 0, 1))
        napari.run()

        # TODO:
        # the ground truth has 3 channels (I am guessing paired with the rgb images)
        # not sure how to handle these cases! need to discuss with CP

        breakpoint()

    return image_paths, gt_paths


def get_m2caiseg_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    assert split in ["train", "val", "test"]

    image_paths, gt_paths = _get_m2caiseg_paths(path=path, split=split, download=download)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_m2caiseg_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_m2caiseg_dataset(
        path=path, split=split, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
