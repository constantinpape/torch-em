"""The CVC-EndoSceneStill dataset contains annotations for polyp segmentation in colonoscopy images.

NOTE: The full CVC-EndoSceneStill release (912 stills split into train / validation / test, with
additional semantic classes for specular highlights and the lumen) is gated behind manual
registration on the CVC-Colon website (https://pages.cvc.uab.es/CVC-Colon/index.php/databases/cvc-endoscenestill/).
We instead provide the openly mirrored "CVC-300" subset, which corresponds to the 60-image test
split of CVC-EndoSceneStill and only ships binary polyp masks. This subset is the one commonly
used as a polyp segmentation benchmark (e.g. in the PraNet line of work).

The dataset is located at https://www.kaggle.com/datasets/nourabentaher/cvc-300.

This dataset is from the publication https://doi.org/10.1155/2017/4037190.
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


KAGGLE_DATASET_NAME = "nourabentaher/cvc-300"


def get_cvc_endoscenestill_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CVC-EndoSceneStill (CVC-300 subset) dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "CVC-300")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name=KAGGLE_DATASET_NAME, download=download)

    zip_path = os.path.join(path, "cvc-300.zip")
    util.unzip(zip_path=zip_path, dst=path)

    if not os.path.exists(data_dir):
        raise RuntimeError(f"The dataset could not be found at '{path}' after extraction.")

    return data_dir


def get_cvc_endoscenestill_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the CVC-EndoSceneStill (CVC-300 subset) data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_cvc_endoscenestill_data(path=path, download=download)

    image_paths = natsorted(glob(os.path.join(data_dir, "images", "*.png")))
    mask_paths = natsorted(glob(os.path.join(data_dir, "masks", "*.png")))

    if len(image_paths) == 0 or len(image_paths) != len(mask_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    neu_gt_dir = os.path.join(data_dir, "masks", "preprocessed")
    os.makedirs(neu_gt_dir, exist_ok=True)

    gt_paths = []
    for mask_path in tqdm(mask_paths, desc="Preprocessing labels"):
        gt_path = os.path.join(neu_gt_dir, f"{Path(mask_path).stem}.tif")
        gt_paths.append(gt_path)
        if os.path.exists(gt_path):
            continue

        mask = imageio.imread(mask_path)
        if mask.ndim == 3:
            mask = np.mean(mask, axis=-1)
        mask = (mask >= 128).astype("uint8")
        imageio.imwrite(gt_path, mask, compression="zlib")

    return image_paths, gt_paths


def get_cvc_endoscenestill_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CVC-EndoSceneStill (CVC-300 subset) dataset for polyp segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_cvc_endoscenestill_paths(path, download)

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


def get_cvc_endoscenestill_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CVC-EndoSceneStill (CVC-300 subset) dataloader for polyp segmentation.

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
    dataset = get_cvc_endoscenestill_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
