"""The SUSTech-SYSU dataset contains 1219 color fundus images from diabetic retinopathy (DR)
patients and healthy controls. 564 of these images have exudate annotations, provided as
Pascal VOC style bounding boxes around hard ('ex') and soft ('se') exudate lesions. These
bounding boxes are rasterized into binary exudate masks here. The dataset also ships DR grades,
optic disc bounding boxes and fovea locations, which are not exposed by this loader.

The dataset is located at https://doi.org/10.6084/m9.figshare.12570770.
This dataset is from the publication https://doi.org/10.1038/s41597-020-00755-0.
Please cite it if you use this dataset in your research.
"""

import os
import xml.etree.ElementTree as ET
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/25320596"
CHECKSUM = "b5e3f31f7fc26f612f5fc04fbc8137a023d4812306ce8d2f92b6011dacd52735"


def get_sustech_sysu_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the SUSTech-SYSU dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "originalImages")
    if os.path.exists(data_dir):
        return path

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "sustech_sysu.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return path


def _bndbox_to_mask(xml_path, mask_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    height, width = int(size.find("height").text), int(size.find("width").text)

    mask = np.zeros((height, width), dtype="uint8")
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        xmin, ymin = int(bndbox.find("xmin").text), int(bndbox.find("ymin").text)
        xmax, ymax = int(bndbox.find("xmax").text), int(bndbox.find("ymax").text)
        mask[ymin:ymax, xmin:xmax] = 1

    imageio.imwrite(mask_path, mask)


def get_sustech_sysu_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the SUSTech-SYSU data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_sustech_sysu_data(path=path, download=download)

    mask_dir = os.path.join(data_dir, "exudatesMasks")
    os.makedirs(mask_dir, exist_ok=True)

    xml_paths = natsorted(glob(os.path.join(data_dir, "exudatesLabels", "*.xml")))

    image_paths, label_paths = [], []
    for xml_path in xml_paths:
        fname = os.path.splitext(os.path.basename(xml_path))[0]
        image_path = os.path.join(data_dir, "originalImages", f"{fname}.jpg")
        assert os.path.exists(image_path), f"The image at '{image_path}' does not exist."

        mask_path = os.path.join(mask_dir, f"{fname}.tif")
        if not os.path.exists(mask_path):
            _bndbox_to_mask(xml_path, mask_path)

        image_paths.append(image_path)
        label_paths.append(mask_path)

    assert len(image_paths) == len(label_paths) and len(image_paths) > 0

    return image_paths, label_paths


def get_sustech_sysu_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SUSTech-SYSU dataset for exudate segmentation in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_sustech_sysu_paths(path=path, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
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


def get_sustech_sysu_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SUSTech-SYSU dataloader for exudate segmentation in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_sustech_sysu_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
