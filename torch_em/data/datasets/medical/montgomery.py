"""The Montgomery dataset contains annotations for lung segmentation
in chest x-ray images.

The database is located at
https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html.
This dataset is from the publication:
- https://doi.org/10.1109/TMI.2013.2284099
- https://doi.org/10.1109/tmi.2013.2290491
Please cite them if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from typing import Union, Tuple, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
CHECKSUM = "54601e952315d8f67383e9202a6e145997ade429f54f7e0af44b4e158714f424"


def get_montgomery_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Montgomery dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "MontgomerySet")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "NLM-MontgomeryCXRSet.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_montgomery_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the Montgomery data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_montgomery_data(path=path, download=download)
    gt_dir = os.path.join(data_dir, "ManualMask", "gt")

    image_paths = sorted(glob(os.path.join(data_dir, "CXR_png", "*.png")))

    if os.path.exists(gt_dir):
        gt_paths = sorted(glob(os.path.join(gt_dir, "*.png")))
        if len(image_paths) == len(gt_paths):
            return image_paths, gt_paths

    else:
        os.makedirs(gt_dir, exist_ok=True)

    lmask_dir = os.path.join(data_dir, "ManualMask", "leftMask")
    rmask_dir = os.path.join(data_dir, "ManualMask", "rightMask")
    gt_paths = []
    for image_path in tqdm(image_paths, desc="Merging left and right lung halves"):
        image_id = os.path.split(image_path)[-1]

        # merge the left and right lung halves into one gt file
        gt = imageio.imread(os.path.join(lmask_dir, image_id))
        gt += imageio.imread(os.path.join(rmask_dir, image_id))
        gt = gt.astype("uint8")

        gt_path = os.path.join(gt_dir, image_id)

        imageio.imwrite(gt_path, gt)
        gt_paths.append(gt_path)

    return image_paths, gt_paths


def get_montgomery_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = True,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Montgomery dataset for lung segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_montgomery_paths(path, download)

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
        **kwargs
    )


def get_montgomery_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = True,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Montgomery dataloader for lung segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_montgomery_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
