"""The PanDental dataset contains annotations for mandible segmentation in
panoramic dental radiographs.

This dataset is located at https://data.mendeley.com/datasets/hxt48yk462/2
This dataset is from the publication https://doi.org/10.1117/1.JMI.2.4.044003.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Literal, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://data.mendeley.com/public-files/datasets/hxt48yk462/files/c2df1e1e-9939-4197-9bac-1eb697a64094/file_downloaded"  # noqa
CHECKSUM = "4ab6f670428df8052ae04aef82011050cd5ad4805f4d28d32fea523880752cf3"


def get_pandental_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PanDental dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Images")
    if os.path.exists(data_dir):
        return path

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "DentalPanoramicXrays.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return path


def get_pandental_paths(
    path: Union[os.PathLike, str], annotator: Literal["1", "2"] = "1", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the PanDental data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        annotator: The choice of expert annotator. Either '1' or '2'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_pandental_data(path=path, download=download)

    image_paths = natsorted(glob(os.path.join(data_dir, "Images", "*.png")))
    raw_gt_paths = natsorted(glob(os.path.join(data_dir, f"Segmentation{annotator}", "*.png")))

    neu_gt_dir = os.path.join(data_dir, "preprocessed", f"gt{annotator}")
    os.makedirs(neu_gt_dir, exist_ok=True)

    gt_paths = []
    for raw_gt_path in tqdm(raw_gt_paths, desc="Preprocessing labels"):
        gt_path = os.path.join(neu_gt_dir, f"{Path(raw_gt_path).stem}.tif")
        gt_paths.append(gt_path)
        if os.path.exists(gt_path):
            continue

        # the ground-truth is the original image masked by the binary mandible region,
        # i.e. non-zero pixels correspond to the mandible.
        raw_gt = imageio.imread(raw_gt_path)
        binary_gt = (raw_gt > 0).astype(np.uint8)
        imageio.imwrite(gt_path, binary_gt)

    return image_paths, gt_paths


def get_pandental_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    annotator: Literal["1", "2"] = "1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PanDental dataset for mandible segmentation in panoramic dental radiographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        annotator: The choice of expert annotator. Either '1' or '2'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_pandental_paths(path, annotator, download)

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
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_pandental_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    annotator: Literal["1", "2"] = "1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PanDental dataloader for mandible segmentation in panoramic dental radiographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        annotator: The choice of expert annotator. Either '1' or '2'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pandental_dataset(path, patch_shape, annotator, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
