import os
from glob import glob
from typing import Union, Tuple, Optional

import json

import torch_em

from .. import util


URL = "https://zenodo.org/records/10390295/files/arcade.zip"
CHECKSUM = "a396cdea7c92c55dc97bbf3dd8e3df517d76872b289a8bcb45513bdb3350837f"


def get_arcade_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "arcade")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "arcade.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _load_annotation_json(json_file):
    assert os.path.exists(json_file)

    with open(json_file, encoding="utf-8") as f:
        gt = json.load(f)

    return gt


def _get_arcade_paths(path, split, task, download):
    data_dir = get_arcade_data(path=path, download=download)

    assert split in ["train", "val", "test"]

    if task is None:
        task = "*"

    image_dirs = sorted(glob(os.path.join(data_dir, task, split, "images")))
    gt_dirs = sorted(glob(os.path.join(data_dir, task, split, "annotations")))

    image_paths, gt_paths = [], []
    for image_dir, gt_dir in zip(image_dirs, gt_dirs):
        json_file = os.path.join(gt_dir, f"{split}.json")
        gt = _load_annotation_json(json_file)

        from collections import defaultdict
        import numpy as np
        import cv2
        import imageio.v3 as imageio

        gt_anns = defaultdict(list)

        for ann in gt["annotations"]:
            gt_anns[ann["image_id"]].append(ann)

        for id, im in gt_anns.items():
            for ann in im:
                points = np.array([ann["segmentation"][0][::2], ann["segmentation"][0][1::2]], np.int32).T
                points = points.reshape(([-1, 1, 2]))
                tmp = np.zeros((512, 512), np.int32)
                cv2.fillPoly(tmp, [points], (1))

                image = imageio.imread(os.path.join(image_dir, f"{id}.png"))

                breakpoint()

                import napari

                v = napari.Viewer()
                v.add_image(image)
                v.add_labels(tmp)
                napari.run()

                breakpoint()

        breakpoint()

    return image_paths, gt_paths


def get_arcade_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: str,
    task: Optional[str] = None,
    download: bool = False,
    **kwargs
):
    image_paths, gt_paths = _get_arcade_paths(path=path, split=split, task=task, download=download)

    dataset = ...

    return dataset


def get_arcade_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: str,
    task: Optional[str] = None,
    download: bool = False,
    **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_arcade_dataset(
        path=path, patch_shape=patch_shape, split=split, task=task, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
