"""The RETA dataset contains annotations for retinal vessel segmentation in fundus images.

The dataset reuses the 81 fundus images from the first (segmentation) subset of the IDRiD dataset
(see `idrid.py`) and adds pixel-level vessel masks obtained with a semi-automated coarse-to-fine
annotation workflow. The public release also ships richer annotations (artery / vein masks,
vascular skeletons, bifurcations, trees and abnormalities) as MATLAB '.mat' files for use with the
authors' own "Computer Aided Retinal Labelling" (CARL) software; these graph / skeleton annotations
are not covered by this module, which only exposes the ready-to-use binary vessel segmentation
masks. Vessel masks for the 27-image test split are withheld by the authors for an online
evaluation server, so only the 54-image training split has publicly available labels.

The dataset is located at https://doi.org/10.6084/m9.figshare.16960855 (CC BY 4.0).
This dataset is from the publication https://doi.org/10.1038/s41597-022-01507-y.
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


URL = "https://ndownloader.figshare.com/files/31398340"
CHECKSUM = "02bd492a252d20c91c4f99f941b54160bd5db8a4b4b061ca272ab5697b818b4f"


def get_reta_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the RETA dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "images")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    rar_path = os.path.join(path, "images.rar")
    util.download_source(path=rar_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip_rarfile(rar_path=rar_path, dst=path)

    return data_dir


def get_reta_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"] = "train", download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the RETA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. The vessel masks for 'test' are withheld by the authors,
            so only 'train' has ground-truth to train / evaluate on.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_reta_data(path=path, download=download)

    assert split in ["train", "test"], f"'{split}' is not a valid split."
    if split == "test":
        raise ValueError(
            "The vessel masks for the 'test' split are withheld by the RETA authors for online "
            "evaluation and are not publicly available. Please use the 'train' split instead."
        )

    image_paths = natsorted(glob(os.path.join(data_dir, split, "img", "*.jpg")))
    raw_gt_paths = natsorted(glob(os.path.join(data_dir, split, "vessel", "*.png")))
    assert len(image_paths) == len(raw_gt_paths) and len(image_paths) > 0

    neu_gt_dir = os.path.join(data_dir, "..", "preprocessed", split)
    os.makedirs(neu_gt_dir, exist_ok=True)

    gt_paths = []
    for image_path, raw_gt_path in tqdm(zip(image_paths, raw_gt_paths), total=len(image_paths), desc="Preprocessing labels"):  # noqa
        assert Path(image_path).stem == Path(raw_gt_path).stem.replace("_vessel", "")

        gt_path = os.path.join(neu_gt_dir, f"{Path(raw_gt_path).stem}.tif")
        gt_paths.append(gt_path)
        if os.path.exists(gt_path):
            continue

        # the masks are stored as (near-)binary 3-channel pngs, i.e. non-zero pixels correspond to
        # vessels in all channels alike. they are binarized into a uint8 (0, 1) single-channel map.
        raw_gt = imageio.imread(raw_gt_path)
        binary_gt = (raw_gt[..., 0] if raw_gt.ndim == 3 else raw_gt) > 0
        imageio.imwrite(gt_path, binary_gt.astype(np.uint8))

    return image_paths, gt_paths


def get_reta_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the RETA dataset for retinal vessel segmentation in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Only 'train' has publicly available vessel masks.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_reta_paths(path, split, download)

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
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_reta_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the RETA dataloader for retinal vessel segmentation in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Only 'train' has publicly available vessel masks.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_reta_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
