"""The ICC dataset contains annotations for interstitial cells of Cajal (ICC) segmentation
in anti-CD117 immunohistochemistry stained anal canal wall images from haemorrhoidal disease
patients.

The dataset is located at https://doi.org/10.5281/zenodo.14900511 under the
CC BY-NC-SA 4.0 license. This dataset is from the publication https://doi.org/10.3390/cells14070550.
Please cite it if you use this dataset in your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "images": "https://zenodo.org/records/14900511/files/Images.zip",
    "annotations": "https://zenodo.org/records/14900511/files/Annotations.coco.json",
}
CHECKSUMS = {
    "images": "d51cb75091891437df80c4ab69412a622c2181ddb9b974ddbc84aeb326542380",
    "annotations": "e98facfdbe04226eb004b40ef06a4c25179ce952ab036d9f00738ceda49708be",
}


def _rasterize_labels(coco, data_dir):
    label_dir = os.path.join(data_dir, "labels")
    if os.path.exists(label_dir) and len(glob(os.path.join(label_dir, "*.tif"))) == len(coco["images"]):
        return label_dir

    os.makedirs(label_dir, exist_ok=True)

    from skimage.draw import polygon as draw_polygon

    annotations_per_image = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_per_image[annotation["image_id"]].append(annotation)

    for image in tqdm(coco["images"], desc="Rasterize the ICC annotations"):
        shape = (image["height"], image["width"])
        labels = np.zeros(shape, dtype="uint16")
        for instance_id, annotation in enumerate(annotations_per_image[image["id"]], start=1):
            polygon = np.array(annotation["segmentation"][0], dtype=float).reshape(-1, 2)
            rows, columns = draw_polygon(polygon[:, 1], polygon[:, 0], shape=shape)
            labels[rows, columns] = instance_id

        name = os.path.splitext(image["file_name"])[0]
        imageio.imwrite(os.path.join(label_dir, f"{name}.tif"), labels, compression="zlib")

    return label_dir


def get_icc_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ICC dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder where the images are stored.
    """
    image_dir = os.path.join(path, "images")
    if os.path.exists(image_dir):
        return path

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "Images.zip")
    util.download_source(path=zip_path, url=URLS["images"], download=download, checksum=CHECKSUMS["images"])
    util.unzip(zip_path=zip_path, dst=image_dir)

    annotation_path = os.path.join(path, "Annotations.coco.json")
    util.download_source(
        path=annotation_path, url=URLS["annotations"], download=download, checksum=CHECKSUMS["annotations"]
    )

    return path


def get_icc_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the ICC data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_icc_data(path, download)

    with open(os.path.join(data_dir, "Annotations.coco.json")) as f:
        coco = json.load(f)

    label_dir = _rasterize_labels(coco, data_dir)

    raw_paths = natsorted(glob(os.path.join(data_dir, "images", "*.jpg")))
    label_paths = natsorted(glob(os.path.join(label_dir, "*.tif")))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(
        os.path.splitext(os.path.basename(raw_path))[0] == os.path.splitext(os.path.basename(label_path))[0]
        for raw_path, label_path in zip(raw_paths, label_paths)
    )

    return raw_paths, label_paths


def get_icc_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the ICC dataset for interstitial cells of Cajal segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_icc_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        ndim=2,
        with_channels=True,
        **kwargs,
    )


def get_icc_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the ICC dataloader for interstitial cells of Cajal segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_icc_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
