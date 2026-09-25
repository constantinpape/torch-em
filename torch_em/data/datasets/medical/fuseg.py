"""The FUSeg dataset contains annotations for wound segmentation in clinical foot ulcer photographs.

The dataset consists of 1,210 foot ulcer images collected over two years from 889 patients for the
2021 MICCAI Foot Ulcer Segmentation (FUSeg) Challenge. Only the 'train' and 'validation' splits are
used here, as they are the only splits with publicly released ground-truth masks; the 'test' split
ground truth is kept private by the challenge organizers for the leaderboard.

The dataset is located at https://github.com/uwm-bigdata/wound-segmentation.
The dataset is from the publication https://doi.org/10.3390/info15030140.
Please cite it if you use this dataset for your research.
"""

import os
import subprocess
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util
from ..light_microscopy.neurips_cell_seg import to_rgb


URL = "https://github.com/uwm-bigdata/wound-segmentation.git"

CHALLENGE_DIR = "Foot Ulcer Segmentation Challenge"


def get_fuseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the FUSeg dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "wound-segmentation", "data", CHALLENGE_DIR)
    if os.path.exists(data_dir):
        return data_dir

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False")

    os.makedirs(path, exist_ok=True)
    repo_dir = os.path.join(path, "wound-segmentation")
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--sparse", "--depth", "1", "--quiet", URL, repo_dir], check=True
    )
    subprocess.run(["git", "sparse-checkout", "set", os.path.join("data", CHALLENGE_DIR)], cwd=repo_dir, check=True)

    return data_dir


def _binarize_mask(mask_path, out_path):
    if os.path.exists(out_path):
        return
    mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 127).astype("uint8")
    imageio.imwrite(out_path, mask)


def get_fuseg_paths(
    path: Union[os.PathLike, str], split: Literal["train", "validation"] = "train", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the FUSeg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'validation'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in ["train", "validation"]:
        raise ValueError(f"'{split}' is not a valid split. Choose 'train' or 'validation'.")

    data_dir = get_fuseg_data(path, download)

    image_dir = os.path.join(data_dir, split, "images")
    label_dir = os.path.join(data_dir, split, "labels")
    preprocessed_dir = os.path.join(data_dir, split, "preprocessed_labels")
    os.makedirs(preprocessed_dir, exist_ok=True)

    image_paths = natsorted(glob(os.path.join(image_dir, "*.png")))

    label_paths = []
    for image_path in image_paths:
        fname = os.path.basename(image_path)
        mask_path = os.path.join(label_dir, fname)
        out_path = os.path.join(preprocessed_dir, fname)
        _binarize_mask(mask_path, out_path)
        label_paths.append(out_path)

    return image_paths, label_paths


def get_fuseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "validation"] = "train",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the FUSeg dataset for wound segmentation in foot ulcer photographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'validation'.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_fuseg_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs,
            patch_shape=patch_shape,
            resize_inputs=resize_inputs,
            resize_kwargs=resize_kwargs,
            ensure_rgb=to_rgb,
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


def get_fuseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "validation"] = "train",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the FUSeg dataloader for wound segmentation in foot ulcer photographs.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'validation'.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_fuseg_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
