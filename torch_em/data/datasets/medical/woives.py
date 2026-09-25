"""The WOIVES dataset contains annotations for retinal vessel segmentation in ultra-widefield
swept-source optical coherence tomography angiography (SS-OCTA) images.

The dataset consists of 206 eyes from 152 participants. Each eye has a superficial-retina OCTA image
(1536x1280 pixels, 24x20 mm field of view) paired with a probabilistic soft-label vessel annotation. The
soft labels are the pixel-wise average of five binary annotations that were made with SAM-OCTA assistance
and then refined by two senior ophthalmologists. They are stored as uint8 images in [0, 255], where
value / 255 is the vessel probability. The release ships no binary masks, and it also contains
deep/full-retina and choroid slabs, fundus photographs and SLO images, which this loader does not use.

By default the soft labels are thresholded into binary vessel masks (see the 'binarize_threshold'
argument). Please report the threshold you use. With 'binarize_threshold=None' the soft probability
maps in [0, 1] are returned as float labels instead.

The official split is a subject-level five-fold cross-validation, selected with the 'fold' and 'split'
arguments.

The data is located at https://doi.org/10.5281/zenodo.21904672, released under a CC-BY-4.0 license.

This dataset is from the publication https://arxiv.org/abs/2609.12574.
Please cite it if you use this dataset for your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List, Optional

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/21904672/files/WOIVES_v1.0.zip/content"
CHECKSUM = "e921284a2b2dc5348607439ead2b567456f73881a5c79251176d700b91f6c757"

SPLITS = ["train", "val", "test"]
FOLDS = [0, 1, 2, 3, 4]


def _convert_labels(label_dir, out_dir, binarize_threshold):
    import tifffile
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    for label_path in tqdm(natsorted(glob(os.path.join(label_dir, "mask_*.png"))), desc="Convert the WOIVES labels"):
        out_path = os.path.join(out_dir, os.path.basename(label_path)[len("mask_"):].replace(".png", ".tif"))
        if os.path.exists(out_path):
            continue

        soft = np.asarray(Image.open(label_path), dtype="float32") / 255.0
        label = soft if binarize_threshold is None else (soft >= binarize_threshold).astype("uint8")

        tmp_path = f"{out_path}.{os.getpid()}.incomplete.tif"
        tifffile.imwrite(tmp_path, label, compression="zlib")
        os.replace(tmp_path, out_path)


def get_woives_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the WOIVES dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the extracted dataset folder.
    """
    data_dir = os.path.join(path, "WOIVES_v1.0")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "WOIVES_v1.0.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path, remove=False)

    assert os.path.exists(data_dir), f"The extraction of the WOIVES archive did not create '{data_dir}'."

    return data_dir


def get_woives_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    fold: int = 0,
    binarize_threshold: Optional[float] = 0.5,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the WOIVES data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. One of 'train', 'val' or 'test'.
        fold: The fold of the official five-fold cross-validation. One of 0 to 4.
        binarize_threshold: The threshold on the vessel probability for creating binary masks (vessel if the
            probability is at least the threshold). Set to None to use the soft probability maps as labels.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose one of {SPLITS}.")
    if fold not in FOLDS:
        raise ValueError(f"'{fold}' is not a valid fold. Choose one of {FOLDS}.")
    if binarize_threshold is not None and not 0.0 < binarize_threshold <= 1.0:
        raise ValueError(f"The threshold must be in (0, 1] or None, but got {binarize_threshold}.")

    data_dir = get_woives_data(path, download)

    label_name = "labels_soft" if binarize_threshold is None else f"labels_bin_{binarize_threshold:g}"
    label_dir = os.path.join(path, label_name)
    _convert_labels(os.path.join(data_dir, "Label"), label_dir, binarize_threshold)

    with open(os.path.join(data_dir, "splits", "fold_split.json")) as f:
        names = json.load(f)["folds"][f"fold{fold}"][split]

    raw_paths = [os.path.join(data_dir, "Image", "OCTA", "OCTA_superficial retina", name) for name in names]
    label_paths = [os.path.join(label_dir, name.replace(".png", ".tif")) for name in names]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in raw_paths + label_paths)

    return raw_paths, label_paths


def get_woives_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    fold: int = 0,
    binarize_threshold: Optional[float] = 0.5,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the WOIVES dataset for retinal vessel segmentation in ultra-widefield OCTA images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. One of 'train', 'val' or 'test'.
        fold: The fold of the official five-fold cross-validation. One of 0 to 4.
        binarize_threshold: The threshold on the vessel probability for creating binary masks.
            Set to None to use the soft probability maps as labels.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_woives_paths(path, split, fold, binarize_threshold, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_woives_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    fold: int = 0,
    binarize_threshold: Optional[float] = 0.5,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the WOIVES dataloader for retinal vessel segmentation in ultra-widefield OCTA images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. One of 'train', 'val' or 'test'.
        fold: The fold of the official five-fold cross-validation. One of 0 to 4.
        binarize_threshold: The threshold on the vessel probability for creating binary masks.
            Set to None to use the soft probability maps as labels.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_woives_dataset(
        path, patch_shape, split, fold, binarize_threshold, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
