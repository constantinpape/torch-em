"""PFUS1 is a dataset for segmentation of pelvic floor anatomical structures in transperineal
ultrasound images, acquired in the midsagittal plane.

The dataset consists of 110 patients (P000-P109), each with several video frames. Every frame is
annotated with polygons for 8 anatomical structures: pubis, urethra, bladder, vagina, uterus, anus,
rectum and levator ani muscle.

This dataset is located at https://doi.org/10.5281/zenodo.10800787 (Zenodo, CC BY 4.0).
The dataset is from the publication https://doi.org/10.1016/j.dib.2025.112346.
Please cite it if you use this dataset for your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
from PIL import Image
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/10800787/files/pfus1.zip"
CHECKSUM = "e65bd7ca941b895ca90c5891b8866ab78bee4575fc7a7c276b9a655456eea380"

LABEL_MAP = {
    "Pubis": 1,
    "Urethra": 2,
    "Bladder": 3,
    "Vagina": 4,
    "Uterus": 5,
    "Anus": 6,
    "Rectum": 7,
    "Levator ani muscle": 8,
}


def get_pfus1_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PFUS1 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "pfus1.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _rasterize_labels(annotation_path, image_path, label_path):
    if os.path.exists(label_path):
        return

    with open(annotation_path) as f:
        annotation = json.load(f)

    from skimage.draw import polygon as draw_polygon

    with Image.open(image_path) as im:
        shape = (im.height, im.width)

    labels = np.zeros(shape, dtype="uint8")
    for shape_annotation in annotation:
        label_id = LABEL_MAP[shape_annotation["label"]]
        points = np.array(shape_annotation["pol"], dtype=float)
        rows, columns = draw_polygon(points[:, 1], points[:, 0], shape=shape)
        labels[rows, columns] = label_id

    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    imageio.imwrite(label_path, labels, compression="zlib")


def get_pfus1_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the PFUS1 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_pfus1_data(path, download)

    image_paths = natsorted(glob(os.path.join(data_dir, "P*", "frame_*.png")))
    assert len(image_paths) > 0

    label_dir = os.path.join(os.path.dirname(data_dir), "labels")
    label_paths = []
    for image_path in tqdm(image_paths, desc="Rasterize the PFUS1 annotations"):
        patient_id = os.path.basename(os.path.dirname(image_path))
        frame_id = os.path.splitext(os.path.basename(image_path))[0]

        annotation_path = os.path.join(data_dir, patient_id, f"{frame_id}.json")
        label_path = os.path.join(label_dir, patient_id, f"{frame_id}.tif")

        _rasterize_labels(annotation_path, image_path, label_path)
        label_paths.append(label_path)

    assert len(image_paths) == len(label_paths)

    return image_paths, label_paths


def get_pfus1_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PFUS1 dataset for segmentation of pelvic floor anatomical structures in ultrasound.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_pfus1_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_pfus1_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PFUS1 dataloader for segmentation of pelvic floor anatomical structures in ultrasound.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pfus1_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
