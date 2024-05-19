import os
from glob import glob
from pathlib import Path
from typing import Union, Tuple

import torch_em
from torch_em.transform.generic import ResizeInputs

from .. import util
from ... import ImageCollectionDataset


TASKS = {
    "microaneurysms": r"1. Microaneurysms",
    "haemorrhages": r"2. Haemorrhages",
    "hard_exudates": r"3. Hard Exudates",
    "soft_exudates": r"4. Soft Exudates",
    "optic_disc": r"5. Optic Disc"
}


def get_idrid_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "data", "A.%20Segmentation")
    if os.path.exists(data_dir):
        return data_dir

    util.download_source_kaggle(
        path=path, dataset_name="aaryapatel98/indian-diabetic-retinopathy-image-dataset", download=download,
    )
    zip_path = os.path.join(path, "indian-diabetic-retinopathy-image-dataset.zip")
    util.unzip(zip_path=zip_path, dst=os.path.join(path, "data"))
    return data_dir


def _get_idrid_paths(path, split, task, download):
    data_dir = get_idrid_data(path=path, download=download)

    split = r"a. Training Set" if split == "train" else r"b. Testing Set"

    gt_paths = sorted(
        glob(
            os.path.join(data_dir, r"A. Segmentation", r"2. All Segmentation Groundtruths", split, TASKS[task], "*.tif")
        )
    )

    image_dir = os.path.join(data_dir, r"A. Segmentation", r"1. Original Images", split)
    image_paths = []
    for gt_path in gt_paths:
        gt_id = Path(gt_path).stem[:-3]
        image_path = os.path.join(image_dir, f"{gt_id}.jpg")
        image_paths.append(image_path)

    return image_paths, gt_paths


def get_idrid_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: str,
    task: str = "optic_disc",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    assert split in ["train", "test"]
    assert task in list(TASKS.keys())

    image_paths, gt_paths = _get_idrid_paths(path=path, split=split, task=task, download=download)

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


def get_idrid_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: str,
    task: str = "optic_disc",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_idrid_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        task=task,
        resize_inputs=resize_inputs,
        download=download,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
