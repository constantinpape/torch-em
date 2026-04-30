"""The Glioma C6 dataset contains phase-contrast microscopy images of Glioma C6
rat brain tumor cells annotated for instance segmentation. It consists of two subsets:

- Glioma C6-spec: 45 images (30 train / 4 val / 11 test) under controlled conditions.
- Glioma C6-gen: 30 images acquired under varied imaging conditions for generalization.

Images are 2592 × 1944 pixels (8-bit TIFF). Annotations are provided in COCO format
with over 20,000 annotated cell and nuclei instances.

The dataset is located at https://zenodo.org/records/15083188.
This dataset is from the following publication:
- Malashin et al. (2025): https://doi.org/10.48550/arXiv.2511.07286
Please cite it if you use this dataset in your research.
"""

import os
import json
from collections import defaultdict
from glob import glob
from natsort import natsorted
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/15083188/files/dataset.zip?download=1"
CHECKSUM = None


def _coco_to_instance_masks(image_dir: str, annotation_file: str, mask_dir: str) -> None:
    """Convert COCO polygon annotations to per-image instance segmentation TIF masks.

    Only cell annotations (supercategory 'cell') are included; nucleus annotations
    (supercategory 'cell_part') are skipped.
    """
    from skimage.draw import polygon as draw_polygon

    with open(annotation_file, "r") as f:
        coco = json.load(f)

    # Keep only cell categories, not cell parts (nuclei).
    cell_cat_ids = {c["id"] for c in coco["categories"] if c.get("supercategory") != "cell_part"}

    images = {img["id"]: img for img in coco["images"]}

    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in cell_cat_ids:
            ann_by_image[ann["image_id"]].append(ann)

    os.makedirs(mask_dir, exist_ok=True)

    for img_id, img_info in images.items():
        fname = img_info["file_name"]
        h, w = img_info["height"], img_info["width"]

        mask = np.zeros((h, w), dtype=np.int32)
        instance_id = 1

        for ann in ann_by_image[img_id]:
            segs = ann.get("segmentation", [])
            if isinstance(segs, dict):
                # RLE format - skip (requires pycocotools)
                continue
            for seg in segs:
                pts = np.array(seg).reshape(-1, 2)
                rr, cc = draw_polygon(pts[:, 1], pts[:, 0], shape=(h, w))
                mask[rr, cc] = instance_id
                instance_id += 1

        mask_name = os.path.splitext(os.path.basename(fname))[0] + "_mask.tif"
        imageio.imwrite(os.path.join(mask_dir, mask_name), mask)


def get_glioma_c6_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Glioma C6 dataset and convert COCO annotations to instance masks.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data directory.
    """
    data_dir = os.path.join(path, "GliomaC6")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(path, "glioma_c6_dataset.zip")
    util.download_source(zip_path, URL, download, checksum=CHECKSUM)
    util.unzip(zip_path, data_dir)

    # Convert COCO annotations to instance masks for each subset/split.
    for ann_file in natsorted(glob(os.path.join(data_dir, "**", "*.json"), recursive=True)):
        subset_dir = os.path.dirname(ann_file)
        image_dir = os.path.join(subset_dir, "images")
        if not os.path.isdir(image_dir):
            image_dir = subset_dir

        split_name = os.path.splitext(os.path.basename(ann_file))[0]
        mask_dir = os.path.join(subset_dir, "masks", split_name)
        _coco_to_instance_masks(image_dir, ann_file, mask_dir)

    return data_dir


def get_glioma_c6_paths(
    path: Union[os.PathLike, str],
    subset: Literal["spec", "gen"] = "spec",
    split: Optional[Literal["train", "val", "test"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Glioma C6 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        subset: The dataset subset. Either 'spec' (controlled, predefined splits) or
            'gen' (generalization, varied conditions).
        split: The data split. One of 'train', 'val', 'test'. Only applies to 'spec'.
            For 'gen', pass None to return all images.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_glioma_c6_data(path, download)
    # Zip extracts as dataset/{subset}/... inside data_dir.
    dataset_dir = os.path.join(data_dir, "dataset", subset)

    if not os.path.isdir(dataset_dir):
        raise RuntimeError(
            f"Could not find '{subset}' subset at {dataset_dir}. "
            "Please check the dataset structure after downloading."
        )

    if subset == "gen":
        image_dir = os.path.join(dataset_dir, "images")
        mask_dir = os.path.join(dataset_dir, "masks", "anno_gen")
        raw_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
        label_paths = natsorted(glob(os.path.join(mask_dir, "*.tif")))
    else:
        # spec subset: each split lives in its own subdirectory.
        # The on-disk directory for "val" is "valid".
        split_dir_name = "valid" if split == "val" else split
        if split_dir_name is None:
            # Return all splits combined.
            raw_paths, label_paths = [], []
            for s, d in [("train", "train"), ("val", "valid"), ("test", "test")]:
                rp, lp = get_glioma_c6_paths(path, subset, s, download)
                raw_paths.extend(rp)
                label_paths.extend(lp)
            return raw_paths, label_paths

        split_dir = os.path.join(dataset_dir, split_dir_name)
        image_dir = os.path.join(split_dir, "images")
        mask_dir = os.path.join(split_dir, "masks", f"anno_{split_dir_name}")
        raw_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
        label_paths = natsorted(glob(os.path.join(mask_dir, "*.tif")))

    if len(raw_paths) == 0:
        raise RuntimeError(f"No images found for subset='{subset}', split='{split}' in {dataset_dir}.")

    return raw_paths, label_paths


def get_glioma_c6_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    subset: Literal["spec", "gen"] = "spec",
    split: Optional[Literal["train", "val", "test"]] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Glioma C6 dataset for phase-contrast cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        subset: The dataset subset. Either 'spec' or 'gen'.
        split: The data split. One of 'train', 'val', 'test' (only for 'spec').
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_glioma_c6_paths(path, subset, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs,
    )


def get_glioma_c6_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    subset: Literal["spec", "gen"] = "spec",
    split: Optional[Literal["train", "val", "test"]] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the Glioma C6 dataloader for phase-contrast cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        subset: The dataset subset. Either 'spec' or 'gen'.
        split: The data split. One of 'train', 'val', 'test' (only for 'spec').
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_glioma_c6_dataset(path, patch_shape, subset, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
