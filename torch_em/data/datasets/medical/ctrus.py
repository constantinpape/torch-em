"""C-TRUS is the Colon Wall Segmentation in Transabdominal Ultrasound dataset, with
annotations for colon wall segmentation in transabdominal ultrasound images of patients
with ulcerative colitis.

The dataset is located at https://github.com/wwu-mmll/c-trus (no explicit license is
stated in the repository).
This dataset is from the publication https://doi.org/10.1007/978-3-031-73647-6_10.
Please cite it if you use this dataset for your research.

NOTE: The labels are stored as JPEG images, so the (originally binary) colon wall masks
have lossy compression artifacts near the mask boundaries. This module binarizes them
with a fixed intensity threshold when caching the labels to disk.
"""

import os
from glob import glob
from tqdm import tqdm
from typing import Union, Tuple, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://github.com/wwu-mmll/c-trus/archive/ad3ce4d4bbc4b89792f757c7aeb83f31f9a229bd.zip"
CHECKSUM = "e584471c8a1340de1ca4180e50469aba1a3ecb554099c5df9aa56f57a9110245"


def get_ctrus_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the C-TRUS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the downloaded images and colon wall annotations.
    """
    data_dir = os.path.join(path, "c-trus-ad3ce4d4bbc4b89792f757c7aeb83f31f9a229bd")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "c-trus.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_ctrus_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the C-TRUS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_ctrus_data(path=path, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "original", "*.jpg")))

    label_dir = os.path.join(data_dir, "labels_binary")
    os.makedirs(label_dir, exist_ok=True)

    gt_paths = []
    for image_path in tqdm(image_paths, desc="Preprocessing C-TRUS labels"):
        fname = os.path.splitext(os.path.basename(image_path))[0]
        gt_path = os.path.join(label_dir, f"{fname}.tif")
        gt_paths.append(gt_path)
        if os.path.exists(gt_path):
            continue

        label_path = os.path.join(data_dir, "labels", f"{fname}.jpg")
        label = imageio.imread(label_path)
        label = (label > 127).astype("uint8")
        imageio.imwrite(gt_path, label, compression="zlib")

    return image_paths, gt_paths


def get_ctrus_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the C-TRUS dataset for colon wall segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_ctrus_paths(path, download)

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
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_ctrus_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the C-TRUS dataloader for colon wall segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ctrus_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
