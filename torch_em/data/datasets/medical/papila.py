import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Union, Tuple, Literal

import numpy as np
from skimage import draw
import imageio.v3 as imageio

import torch_em

from .. import util


URL = "https://figshare.com/ndownloader/files/35013982"
CHECKSUM = "15b053dff496bc8e53eb8a8d0707ef73ba3d56c988eea92b65832c9c82852a7d"


def get_papila_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "papila.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


# contour_to_mask() functions taken from https://github.com/matterport/Mask_RCNN
def contour_to_mask(cont, img_shape):
    """Return mask given a contour and the shape of image
    """
    c = np.loadtxt(cont)
    mask = np.zeros(img_shape[:-1], dtype=np.uint8)
    rr, cc = draw.polygon(c[:, 1], c[:, 0])
    mask[rr, cc] = 1
    return mask


def _get_papila_paths(path, task, expert_choice, download):
    data_dir = get_papila_data(path=path, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "FundusImages", "*.jpg")))

    gt_dir = os.path.join(data_dir, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)

    patient_ids = [Path(image_path).stem for image_path in image_paths]

    input_shape = (1934, 2576, 3)  # shape of the input images
    gt_paths = []
    for patient_id in tqdm(patient_ids, desc=f"Converting contours to segmentations for '{expert_choice}'"):
        gt_contours = sorted(
            glob(os.path.join(data_dir, "ExpertsSegmentations", "Contours", f"{patient_id}_{task}_{expert_choice}.txt"))
        )

        for gt_contour in gt_contours:
            tmp_task = Path(gt_contour).stem.split("_")[1]
            gt_path = os.path.join(gt_dir, f"{patient_id}_{tmp_task}_{expert_choice}.tif")
            gt_paths.append(gt_path)
            if os.path.exists(gt_path):
                continue

            semantic_labels = contour_to_mask(cont=gt_contour, img_shape=input_shape)
            imageio.imwrite(gt_path, semantic_labels)

    return image_paths, gt_paths


def get_papila_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    task: Literal["cup", "disc"] = "disc",
    expert_choice: Literal["exp1", "exp2"] = "exp1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of optic cup and optic disc in fundus images.

    The database is located at https://figshare.com/articles/dataset/PAPILA/14798004/2

    The dataset is from Kovalyk et al. - https://doi.org/10.1038/s41597-022-01388-1.
    Please cite it if you use this dataset for a publication.
    """
    assert expert_choice in ["exp1", "exp2"], f"'{expert_choice}' is not a valid expert choice."
    assert task in ["cup", "disc"], f"'{task}' is not a valid task."

    image_paths, gt_paths = _get_papila_paths(path=path, task=task, expert_choice=expert_choice, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
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


def get_papila_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    task: Literal["cup", "disc"] = "disc",
    expert_choice: Literal["exp1", "exp2"] = "exp1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of optic cup and optic disc in fundus images. See `get_papila_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_papila_dataset(
        path=path,
        patch_shape=patch_shape,
        task=task,
        expert_choice=expert_choice,
        resize_inputs=resize_inputs,
        download=download,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
