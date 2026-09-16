"""The US Simulation & Segmentation dataset contains real and simulated abdominal ultrasound
scans with manual segmentations of several abdominal organs.

The real scans ('rus') were acquired from 11 healthy subjects with a SonoSite M Turbo V1.3
ultrasound device. The simulated scans ('aus') were generated with a ray-casting based
simulator from CT volumes of the VISCERAL Anatomy3 challenge. Only a subset of the images
in each domain has manual (real) or silver-standard (simulated) annotations of the liver,
kidney, pancreas, vessels, adrenals, gallbladder, bones and spleen.

The dataset is located at https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm.
This dataset is from the publication https://doi.org/10.1007/s11548-019-02046-5.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util
from ..light_microscopy.neurips_cell_seg import to_rgb


LABEL_MAPS = {
    (0, 0, 0): 0,  # background (outside the field of view)
    (10, 10, 10): 0,  # background tissue (only present in the simulated annotations)
    (100, 0, 100): 1,  # liver (violet)
    (255, 255, 0): 2,  # kidney (yellow)
    (0, 0, 255): 3,  # pancreas (blue)
    (255, 0, 0): 4,  # vessels (red)
    (0, 255, 255): 5,  # adrenals (light blue)
    (0, 255, 0): 6,  # gallbladder (green)
    (255, 255, 255): 7,  # bones (white)
    (255, 0, 255): 8,  # spleen (pink)
}


def get_us_sim_and_seg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the US Simulation & Segmentation dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "abdominal_US", "abdominal_US")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "ussimandsegm.zip")
    util.download_source_kaggle(path=path, dataset_name="ignaciorlando/ussimandsegm", download=download)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _preprocess_labels(image_paths, gt_paths, gt_dir):
    os.makedirs(gt_dir, exist_ok=True)

    neu_gt_paths = []
    for image_path, gt_path in tqdm(
        zip(image_paths, gt_paths), total=len(image_paths), desc="Preprocessing labels"
    ):
        neu_gt_path = os.path.join(gt_dir, f"{Path(image_path).stem}.tif")
        neu_gt_paths.append(neu_gt_path)
        if os.path.exists(neu_gt_path):
            continue

        gt = imageio.imread(gt_path)
        if gt.ndim == 2:
            gt = np.stack([gt] * 3, axis=-1)
        gt = gt[..., :3]  # some annotations have an additional (constant) alpha channel.

        labels = np.zeros(gt.shape[:2], dtype="uint8")
        for color, label_id in LABEL_MAPS.items():
            if label_id == 0:
                continue
            binary_map = (gt == color).all(axis=-1)
            labels[binary_map] = label_id

        imageio.imwrite(neu_gt_path, labels, compression="zlib")

    return neu_gt_paths


def get_us_sim_and_seg_paths(
    path: Union[os.PathLike, str],
    source: Literal["aus", "rus"],
    split: Literal["train", "test"],
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the US Simulation & Segmentation data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        source: The choice of data source. Either 'aus' (simulated scans) or 'rus' (real scans).
        split: The choice of data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if source not in ("aus", "rus"):
        raise ValueError(f"'{source}' is not a valid source. Choose either 'aus' or 'rus'.")

    if split not in ("train", "test"):
        raise ValueError(f"'{split}' is not a valid split. Choose either 'train' or 'test'.")

    data_dir = get_us_sim_and_seg_data(path, download)

    image_dir = os.path.join(data_dir, source.upper(), "images", split)
    gt_dir = os.path.join(data_dir, source.upper(), "annotations", split)

    if not os.path.exists(gt_dir):
        raise ValueError(f"There are no annotations for the '{source}' source and the '{split}' split.")

    image_paths = natsorted(glob(os.path.join(image_dir, "*")))
    gt_paths = natsorted(glob(os.path.join(gt_dir, "*")))

    image_stems = {Path(p).stem: p for p in image_paths}
    gt_stems = {Path(p).stem: p for p in gt_paths}
    matched_stems = natsorted(set(image_stems) & set(gt_stems))
    assert len(matched_stems) > 0

    image_paths = [image_stems[stem] for stem in matched_stems]
    gt_paths = [gt_stems[stem] for stem in matched_stems]

    neu_gt_dir = os.path.join(data_dir, source.upper(), "preprocessed", split)
    neu_gt_paths = _preprocess_labels(image_paths, gt_paths, neu_gt_dir)

    return image_paths, neu_gt_paths


def get_us_sim_and_seg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    source: Literal["aus", "rus"],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the US Simulation & Segmentation dataset for abdominal organ segmentation in ultrasound images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        source: The choice of data source. Either 'aus' (simulated scans) or 'rus' (real scans).
        split: The choice of data split. Either 'train' or 'test'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_us_sim_and_seg_paths(path, source, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs,
            patch_shape=patch_shape,
            resize_inputs=resize_inputs,
            resize_kwargs=resize_kwargs,
            ensure_rgb=to_rgb,
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


def get_us_sim_and_seg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    source: Literal["aus", "rus"],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the US Simulation & Segmentation dataloader for abdominal organ segmentation in ultrasound images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        source: The choice of data source. Either 'aus' (simulated scans) or 'rus' (real scans).
        split: The choice of data split. Either 'train' or 'test'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_us_sim_and_seg_dataset(path, patch_shape, source, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
