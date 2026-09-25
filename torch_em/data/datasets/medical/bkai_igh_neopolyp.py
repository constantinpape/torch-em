"""The BKAI-IGH NeoPolyp dataset contains annotations for semantic segmentation of neoplastic
and non-neoplastic polyps in colonoscopy images.

NOTE: The ground-truth masks are stored as red (neoplastic polyp) and green (non-neoplastic
polyp) regions on a black background, and the archive stores them as JPEG images. This means
that lossy compression introduces off-palette colors along the region boundaries. We resolve
this by assigning each pixel to whichever of the three reference colors (background, red,
green) it is closest to, yielding semantic labels: 0 (background), 1 (non-neoplastic polyp)
and 2 (neoplastic polyp).

NOTE: This dataset requires the Kaggle API. You need to install it via 'pip install kaggle'
and set up an API token, see https://www.kaggle.com/docs/api. You also need to accept the
competition rules on the Kaggle website (https://www.kaggle.com/c/bkai-igh-neopolyp/rules)
before the download will succeed.

The dataset is located at https://www.kaggle.com/c/bkai-igh-neopolyp.
This dataset is from the publication https://doi.org/10.1007/978-3-030-90436-4_2.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


LABEL_COLORS = {
    0: (0, 0, 0),  # background
    1: (0, 255, 0),  # non-neoplastic polyp
    2: (255, 0, 0),  # neoplastic polyp
}


def get_bkai_igh_neopolyp_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BKAI-IGH NeoPolyp dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "train")
    if os.path.exists(data_dir):
        return path

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "bkai-igh-neopolyp.zip")
    util.download_source_kaggle(path=path, dataset_name="bkai-igh-neopolyp", download=download, competition=True)
    util.unzip(zip_path=zip_path, dst=path)

    # The competition bundle ships 'train', 'train_gt' and 'test' as nested zip archives.
    for name in ["train", "train_gt", "test"]:
        nested_zip = os.path.join(path, f"{name}.zip")
        if os.path.exists(nested_zip):
            util.unzip(zip_path=nested_zip, dst=path)

    return path


def get_bkai_igh_neopolyp_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the BKAI-IGH NeoPolyp data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_bkai_igh_neopolyp_data(path=path, download=download)

    image_paths = natsorted(glob(os.path.join(data_dir, "train", "train", "*.jpeg")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "train_gt", "train_gt", "*.jpeg")))

    neu_gt_dir = os.path.join(data_dir, "train_gt", "preprocessed")
    os.makedirs(neu_gt_dir, exist_ok=True)

    reference_colors = np.array(list(LABEL_COLORS.values()))

    neu_gt_paths = []
    for gt_path in tqdm(gt_paths, desc="Preprocessing labels"):
        neu_gt_path = os.path.join(neu_gt_dir, f"{Path(gt_path).stem}.tif")
        neu_gt_paths.append(neu_gt_path)
        if os.path.exists(neu_gt_path):
            continue

        gt = imageio.imread(gt_path)[..., :3].astype("float32")
        distances = np.linalg.norm(gt[..., None, :] - reference_colors[None, None, :, :], axis=-1)
        semantic_gt = np.argmin(distances, axis=-1).astype("uint8")
        imageio.imwrite(neu_gt_path, semantic_gt, compression="zlib")

    return image_paths, neu_gt_paths


def get_bkai_igh_neopolyp_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BKAI-IGH NeoPolyp dataset for polyp segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_bkai_igh_neopolyp_paths(path, download)

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


def get_bkai_igh_neopolyp_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BKAI-IGH NeoPolyp dataloader for polyp segmentation.

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
    dataset = get_bkai_igh_neopolyp_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
