"""FASS is the Fetal Abdominal Structures Segmentation dataset for segmenting the
abdominal aorta artery, intrahepatic umbilical vein, stomach, and liver in fetal
abdominal circumference ultrasound images.

The dataset is located at https://data.mendeley.com/datasets/4gcpm9dsc3/1 (CC BY 4.0).
This dataset is from the publication https://doi.org/10.17632/4gcpm9dsc3.1.
Please cite it if you use this dataset for your research.

NOTE: Mendeley's "Download all files" link goes through a Cloudflare bot check that
blocks plain HTTP clients, so this module downloads the dataset's zip archive via its
`public-files` mirror url instead, which is not gated behind that check.
"""

import os
from glob import glob
from tqdm import tqdm
from typing import Union, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://data.mendeley.com/public-files/datasets/4gcpm9dsc3/files/89e74076-ff57-4e81-9634-4fc29c6128ff/file_downloaded"  # noqa
CHECKSUM = "bde4ab58689bdbc59fa8c0222d41a9fbe1edf16dab40ac95346b6c2f2a7d22ee"

# The order in which the (mostly non-overlapping) per-structure binary masks are painted
# into the single-channel label. A structure painted later wins ties on the rare pixels
# where two structures' masks overlap.
STRUCTURES = ["liver", "stomach", "artery", "vein"]


def get_fass_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the FASS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the downloaded images and structure annotations.
    """
    data_dir = os.path.join(path, "ARRAY_FORMAT")
    if os.path.exists(data_dir):
        return path

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "fass.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return path


def get_fass_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the FASS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_fass_data(path=path, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "IMAGES", "*.png")))

    label_dir = os.path.join(data_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)

    gt_paths = []
    for image_path in tqdm(image_paths, desc="Preprocessing FASS labels"):
        fname = os.path.splitext(os.path.basename(image_path))[0]
        gt_path = os.path.join(label_dir, f"{fname}.tif")
        gt_paths.append(gt_path)
        if os.path.exists(gt_path):
            continue

        npy_path = os.path.join(data_dir, "ARRAY_FORMAT", f"{fname}.npy")
        structures = np.load(npy_path, allow_pickle=True).item()["structures"]

        label = np.zeros(structures["liver"].shape, dtype="uint8")
        for i, name in enumerate(STRUCTURES, start=1):
            label[structures[name] > 0] = i

        imageio.imwrite(gt_path, label, compression="zlib")

    return image_paths, gt_paths


def get_fass_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the FASS dataset for fetal abdominal structure segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_fass_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
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


def get_fass_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the FASS dataloader for fetal abdominal structure segmentation.

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
    dataset = get_fass_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
