"""The ARCADE dataset contains annotations for coronary vessel segmentation in
X-Ray Coronary Angiograms.

The dataset is from the challenge: https://doi.org/10.1038/s41597-023-02871-z.
The dataset is located at: https://doi.org/10.5281/zenodo.10390295.
Please cite them if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from collections import defaultdict
from typing import Union, Tuple, Literal, List

import json
import numpy as np
import imageio.v3 as imageio

import torch_em

from .. import util


URL = "https://zenodo.org/records/10390295/files/arcade.zip"
CHECKSUM = "a396cdea7c92c55dc97bbf3dd8e3df517d76872b289a8bcb45513bdb3350837f"


def get_arcade_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ARCADE dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "arcade")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "arcade.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _load_annotation_json(json_file):
    assert os.path.exists(json_file)

    with open(json_file, encoding="utf-8") as f:
        gt_ann_json_file = json.load(f)

    return gt_ann_json_file


def get_arcade_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'val', 'test'],
    task: Literal['syntax'] = "syntax",
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the ARCADE data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'train', 'val' or 'test'.
        task: The choice of task. By default, 'syntax'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    import cv2

    data_dir = get_arcade_data(path=path, download=download)

    assert split in ["train", "val", "test"]

    if task is None:
        task = "*"

    image_dirs = natsorted(glob(os.path.join(data_dir, task, split, "images")))
    gt_dirs = natsorted(glob(os.path.join(data_dir, task, split, "annotations")))

    image_paths, gt_paths = [], []
    for image_dir, gt_dir in zip(image_dirs, gt_dirs):
        json_file = os.path.join(gt_dir, f"{split}.json")
        gt = _load_annotation_json(json_file)

        # THE RECOMMENDED WAY FROM THE DATA PROVIDERS TO CONVERT FROM COCO TO MASKS
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

            semantic_labels = np.zeros((512, 512), np.int32)  # NOTE: The input shapes are known.
            for ann in im:
                points = np.array([ann["segmentation"][0][::2], ann["segmentation"][0][1::2]], np.int32).T
                points = points.reshape(([-1, 1, 2]))
                tmp = np.zeros((512, 512), np.int32)
                cv2.fillPoly(semantic_labels, [points], (1))
                semantic_labels += tmp

            imageio.imwrite(gt_path, semantic_labels)

    breakpoint()

    return image_paths, gt_paths


def get_arcade_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    task: Literal['syntax'] = "syntax",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Get the ARCADE dataset for coronary artery segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        task: The choice of task. By default, 'syntax'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    # TODO: the "stenosis" data has 3 channels, the "syntax" data has 1 channel
    # for us, the relevant one is the "syntax" task, as we are interest in segmenting vessels for our workflows.
    # for the "stenosis" task, the segmentations are only for the
    # "stenotic valves" (i.e. abnormal narrowing of a certain region of the arteries)
    image_paths, gt_paths = get_arcade_paths(path, split, task, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_arcade_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: Literal['train', 'val', 'test'],
    task: Literal['syntax'] = "syntax",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Get the ARCADE dataloader for coronary artery segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        task: The choice of task. By default, 'syntax'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_arcade_dataset(path, patch_shape, split, task, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
