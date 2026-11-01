"""The BBBC041 dataset contains brightfield images of Giemsa-stained blood smears from patients with
P. vivax malaria, with bounding box annotations for each cell and its class.

The dataset has 1,328 images: 1,208 for training and 120 for testing. Each cell is annotated with a box and
one of these classes: red blood cell, trophozoite, ring, schizont, gametocyte, leukocyte and difficult.
This loader draws the boxes into label images, so the instance and semantic labels have box shapes:
- 'instances': every box gets its own id.
- 'semantic': every box gets the id of its class, see CLASSES.
Boxes overlap in dense smears. The larger boxes are drawn first, so that a smaller box stays visible on top.

The dataset is located at https://bbbc.broadinstitute.org/BBBC041.
This dataset is from the publication https://doi.org/10.1007/978-3-319-66179-7_1.
Please cite it if you use this dataset in your research.
"""

import os
import json
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://data.broadinstitute.org/bbbc/BBBC041/malaria.zip"
CHECKSUM = "46e47b02284fcab4d3f8cff291f08f0dbc1e9296c4ec974f3f4d989652446e81"

# The semantic id of each class. The id 0 is the background.
CLASSES = {
    "red blood cell": 1,
    "trophozoite": 2,
    "ring": 3,
    "schizont": 4,
    "gametocyte": 5,
    "leukocyte": 6,
    "difficult": 7,
}

SPLIT_FILES = {"train": "training.json", "test": "test.json"}


def _draw_boxes(objects, shape):
    """Draw the boxes of the objects into an instance and a semantic label image, the largest boxes first."""
    def area(obj):
        box = obj["bounding_box"]
        return (box["maximum"]["r"] - box["minimum"]["r"]) * (box["maximum"]["c"] - box["minimum"]["c"])

    instances = np.zeros(shape, dtype="int64")
    semantic = np.zeros(shape, dtype="uint8")
    for instance_id, obj in enumerate(sorted(objects, key=area, reverse=True), start=1):
        box = obj["bounding_box"]
        r0, c0 = max(box["minimum"]["r"], 0), max(box["minimum"]["c"], 0)
        r1, c1 = min(box["maximum"]["r"], shape[0]), min(box["maximum"]["c"], shape[1])
        instances[r0:r1, c0:c1] = instance_id
        semantic[r0:r1, c0:c1] = CLASSES[obj["category"]]
    return instances, semantic


def _create_h5_data(data_dir, h5_dir, split):
    """Create one h5 file per image with the raw image and the instance and semantic labels."""
    import h5py
    from tqdm import tqdm

    os.makedirs(h5_dir, exist_ok=True)
    with open(os.path.join(data_dir, SPLIT_FILES[split])) as f:
        annotations = json.load(f)

    for annotation in tqdm(annotations, desc=f"Creating h5 files for '{split}'"):
        image_path = os.path.join(data_dir, annotation["image"]["pathname"].lstrip("/"))
        h5_path = os.path.join(h5_dir, os.path.splitext(os.path.basename(image_path))[0] + ".h5")
        if os.path.exists(h5_path):
            continue

        raw = imageio.imread(image_path)
        if raw.ndim == 3:
            raw = raw[..., :3].transpose(2, 0, 1)
        instances, semantic = _draw_boxes(annotation["objects"], raw.shape[-2:])

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/instances", data=instances, compression="gzip")
            f.create_dataset("labels/semantic", data=semantic, compression="gzip")


def get_bbbc041_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BBBC041 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the images and the annotation files.
    """
    data_dir = os.path.join(path, "malaria")
    if os.path.exists(os.path.join(data_dir, SPLIT_FILES["train"])):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "malaria.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_bbbc041_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> List[str]:
    """Get paths to the BBBC041 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the h5 data.
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"'{split}' is not a valid split. Choose either 'train' or 'test'.")

    data_dir = get_bbbc041_data(path, download)
    h5_dir = os.path.join(path, "h5_data", split)
    with open(os.path.join(data_dir, SPLIT_FILES[split])) as f:
        n_images = len(json.load(f))
    if len(glob(os.path.join(h5_dir, "*.h5"))) < n_images:
        _create_h5_data(data_dir, h5_dir, split)

    h5_paths = natsorted(glob(os.path.join(h5_dir, "*.h5")))
    assert len(h5_paths) == n_images, f"Expected {n_images} h5 files for '{split}', found {len(h5_paths)}."
    return h5_paths


def get_bbbc041_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    label_choice: Literal["instances", "semantic"] = "instances",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BBBC041 dataset for blood cell segmentation and classification in malaria blood smears.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        label_choice: The choice of labels. Either 'instances' (one id per box) or 'semantic' (the class of each
            box, see CLASSES). Both are drawn from the bounding boxes.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if label_choice not in ("instances", "semantic"):
        raise ValueError(f"'{label_choice}' is not a valid label choice. Choose either 'instances' or 'semantic'.")

    h5_paths = get_bbbc041_paths(path, split, download)

    if label_choice == "instances":
        kwargs, _ = util.add_instance_label_transform(kwargs, add_binary_target=True, label_dtype=np.int64)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=h5_paths,
        raw_key="raw",
        label_paths=h5_paths,
        label_key=f"labels/{label_choice}",
        patch_shape=patch_shape,
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_bbbc041_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    label_choice: Literal["instances", "semantic"] = "instances",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BBBC041 dataloader for blood cell segmentation and classification in malaria blood smears.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        label_choice: The choice of labels. Either 'instances' (one id per box) or 'semantic' (the class of each
            box, see CLASSES). Both are drawn from the bounding boxes.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bbbc041_dataset(path, patch_shape, split, label_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
