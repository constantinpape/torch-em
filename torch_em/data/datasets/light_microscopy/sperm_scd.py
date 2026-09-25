"""The SCD Sperm dataset contains annotations for sperm instance segmentation in bright-field microscopy
images of the sperm chromatin dispersion (SCD) assay, which is used to assess sperm DNA fragmentation.

The images (1920x1080, RGB) were acquired from 40 coded slides (L01 to L40) and every sperm cell is annotated
with a polygon and one of two classes: fragmented (halo-less) or non-fragmented (with halo). The official splits
are defined per slide: L01-L32 for 'train', L33-L36 for 'val' and L37-L40 for 'test'. This module rasterizes
the COCO polygons into instance labels (one id per sperm, starting from 1) or into class labels
(1 = fragmented, 2 = non-fragmented). If polygons overlap, the smaller instance is painted last and wins.

NOTE: The release only ships the images of a subset of the slides (about 1,100 of the 2,000 images described in its
README), while the annotations are provided for all slides. This module only uses annotated images that are
present on disk, which are discovered from the files, and it ignores the images without annotated sperm.

The dataset is located at https://doi.org/10.5281/zenodo.21628868 and is released under a CC-BY-4.0 license.
Please cite it if you use this dataset for your research.
"""

import os
import json
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/21628868/files/SCD_Microscopy_YOLOv8seg_Repository.zip/content"
CHECKSUM = "6e8d9e03f9037c7316113572497fc5ac61ddc2dc63f504b4af91a01c0ca1afeb"

SPLITS = ["train", "val", "test"]
LABEL_TYPES = ["instances", "classes"]


def _slide_splits(root):
    slide_to_split = {}
    for split in SPLITS:
        for label_path in glob(os.path.join(root, "dataset", "labels_split", split, "*.txt")):
            slide_to_split[os.path.basename(label_path)[:3]] = split
    return slide_to_split


def _find_images(root):
    image_paths = {}
    for pattern in (("dataset", "images", "*.png"), ("dataset", "images_split", "*", "*.png")):
        for image_path in glob(os.path.join(root, *pattern)):
            image_paths.setdefault(os.path.basename(image_path), image_path)
    return image_paths


def _rasterize(annotations, shape):
    from skimage.draw import polygon

    instances = np.zeros(shape, dtype="uint16")
    classes = np.zeros(shape, dtype="uint8")
    for instance_id, annotation in enumerate(sorted(annotations, key=lambda a: -a["area"]), start=1):
        for part in annotation["segmentation"]:
            coords = np.asarray(part, dtype="float64").reshape(-1, 2)
            rr, cc = polygon(coords[:, 1], coords[:, 0], shape=shape)
            instances[rr, cc] = instance_id
            classes[rr, cc] = annotation["category_id"]
    return instances, classes


def _write_tif(path, array):
    import tifffile

    tmp_path = f"{path}.{os.getpid()}.incomplete.tif"
    tifffile.imwrite(tmp_path, array, compression="zlib")
    os.replace(tmp_path, path)


def _preprocess_labels(root, label_root):
    done_marker = os.path.join(label_root, "done")
    if os.path.exists(done_marker):
        return

    image_paths = _find_images(root)

    per_image = {}
    for json_path in natsorted(glob(os.path.join(root, "annotations", "coco_json", "*.json"))):
        with open(json_path) as f:
            coco = json.load(f)
        file_names = {image["id"]: image for image in coco["images"]}
        for annotation in coco["annotations"]:
            image = file_names[annotation["image_id"]]
            per_image.setdefault(image["file_name"], (image, []))[1].append(annotation)

    for label_type in LABEL_TYPES:
        os.makedirs(os.path.join(label_root, label_type), exist_ok=True)

    for file_name, (image, annotations) in tqdm(sorted(per_image.items()), desc="Preprocess SCD Sperm"):
        if file_name not in image_paths:
            continue

        stem = os.path.splitext(file_name)[0]
        out_paths = [os.path.join(label_root, label_type, f"{stem}.tif") for label_type in LABEL_TYPES]
        if all(os.path.exists(p) for p in out_paths):
            continue

        instances, classes = _rasterize(annotations, (image["height"], image["width"]))
        for out_path, array in zip(out_paths, (instances, classes)):
            _write_tif(out_path, array)

    with open(done_marker, "w"):
        pass


def get_sperm_scd_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the SCD Sperm dataset and rasterize its annotations.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the extracted data.
    """
    root = os.path.join(path, "SCD_Microscopy_YOLOv8seg_Repository")
    if not os.path.exists(root):
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, "SCD_Microscopy_YOLOv8seg_Repository.zip")
        util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip(zip_path=zip_path, dst=path, remove=False)

    _preprocess_labels(root, os.path.join(path, "labels"))
    return root


def get_sperm_scd_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    label_type: Literal["instances", "classes"] = "instances",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the SCD Sperm data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. One of 'train', 'val' or 'test'.
        label_type: The choice of labels. Either 'instances' (one id per sperm) or 'classes'
            (1 = fragmented, 2 = non-fragmented).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose one of {SPLITS}.")
    if label_type not in LABEL_TYPES:
        raise ValueError(f"'{label_type}' is not a valid label type. Choose one of {LABEL_TYPES}.")

    root = get_sperm_scd_data(path, download)
    slide_to_split = _slide_splits(root)
    image_paths = _find_images(root)

    raw_paths, label_paths = [], []
    for label_path in natsorted(glob(os.path.join(path, "labels", label_type, "*.tif"))):
        stem = os.path.splitext(os.path.basename(label_path))[0]
        if slide_to_split.get(stem[:3]) == split and f"{stem}.png" in image_paths:
            raw_paths.append(image_paths[f"{stem}.png"])
            label_paths.append(label_path)

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    return raw_paths, label_paths


def get_sperm_scd_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    label_type: Literal["instances", "classes"] = "instances",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SCD Sperm dataset for sperm instance segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. One of 'train', 'val' or 'test'.
        label_type: The choice of labels. Either 'instances' (one id per sperm) or 'classes'
            (1 = fragmented, 2 = non-fragmented).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_sperm_scd_paths(path, split, label_type, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_sperm_scd_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    label_type: Literal["instances", "classes"] = "instances",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SCD Sperm dataloader for sperm instance segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. One of 'train', 'val' or 'test'.
        label_type: The choice of labels. Either 'instances' (one id per sperm) or 'classes'
            (1 = fragmented, 2 = non-fragmented).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_sperm_scd_dataset(path, patch_shape, split, label_type, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
