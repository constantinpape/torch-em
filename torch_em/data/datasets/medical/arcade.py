import os
from glob import glob
from tqdm import tqdm
from typing import Union, Tuple
from collections import defaultdict

import cv2
import json
import numpy as np
import imageio.v3 as imageio

import torch_em
from torch_em.transform.generic import ResizeInputs

from .. import util
from ... import ImageCollectionDataset


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
        gt_ann_json_file = json.load(f)

    return gt_ann_json_file


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

        # THE RECOMMENDED WAY FROM THE DATA PROVIDERS TO CONVERT FROM COCO TO MASKS #

        gt_anns = defaultdict(list)

        for ann in gt["annotations"]:
            gt_anns[ann["image_id"]].append(ann)

        for id, im in tqdm(gt_anns.items(), desc="Creating ARCADE segmentations from coco-style annotations"):
            image_path = os.path.join(image_dir, f"{id}.png")
            gt_path = os.path.join(gt_dir, f"{id}.tif")

            image_paths.append(image_path)
            gt_paths.append(gt_path)

            if os.path.exists(gt_path):
                continue

            semantic_labels = np.zeros((512, 512), np.int32)
            for ann in im:
                points = np.array([ann["segmentation"][0][::2], ann["segmentation"][0][1::2]], np.int32).T
                points = points.reshape(([-1, 1, 2]))
                tmp = np.zeros((512, 512), np.int32)
                cv2.fillPoly(semantic_labels, [points], (1))
                semantic_labels += tmp

            imageio.imwrite(gt_path, semantic_labels)

        # DESIRED WAY #
        # issues: the method does work quite nicely, however it does not work for some image ids (e.g. 1, 2, 922)
        # while works for others (923, 924)

        # from pycocotools.coco import COCO
        # import numpy as np
        # import imageio.v3 as imageio

        # coco = COCO(json_file)

        # image_ids = coco.getImgIds()
        # image_id = 925  # image_ids[0]
        # image_metadata = coco.loadImgs(image_id)[0]
        # fname = image_metadata["file_name"]

        # ann_ids = coco.getAnnIds(imgIds=image_metadata["id"])
        # anns = coco.loadAnns(ann_ids)

        # shape = (image_metadata["height"], image_metadata["width"])
        # seg = np.zeros(shape, dtype="uint32")

        # for ann in anns:
        #     mask = coco.annToMask(ann).astype("bool")
        #     seg[mask] = 1

        # image = imageio.imread(os.path.join(image_dir, fname))

        # import napari

        # v = napari.Viewer()
        # v.add_image(image)
        # v.add_image(seg > 0)
        # napari.run()

        # breakpoint()

    return image_paths, gt_paths


def get_arcade_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: str,
    task: str = "syntax",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """
    """
    # TODO: the "stenosis" data has 3 channels, the "syntax" data has 1 channel
    # for us, the relevant one is the "syntax" task, as we are interest in segmenting vessels for our workflows.
    # for the "stenosis" task, the segmentations are only for the
    # "stenotic valves" (i.e. abnormal narrowing of a certain region of the arteries)
    image_paths, gt_paths = _get_arcade_paths(path=path, split=split, task=task, download=download)

    if resize_inputs:
        raw_trafo = ResizeInputs(target_shape=patch_shape, is_label=False)
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


def get_arcade_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: str,
    task: str = "syntax",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_arcade_dataset(
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
