"""CT2US for Kidney Segmentation is a dataset that contains synthetic ultrasound images of
kidneys with their corresponding kidney segmentation masks.

The synthetic ultrasound images are generated from annotated CT slices via CycleGAN-based
cross-modal style transfer, in order to construct a large "transition" dataset for training
segmentation models on ultrasound data despite the scarcity of annotated real ultrasound
kidney images. NOTE: The Kaggle package for this dataset only ships the resulting synthetic
ultrasound slices (`slice`) paired with their kidney masks (`mask`); the source CT volumes
and the CycleGAN model are not part of this package. We therefore use the synthetic
ultrasound images with their masks for the segmentation loader below.

The dataset is located at https://www.kaggle.com/datasets/siatsyx/ct2usforkidneyseg.
The corresponding code repository is at https://github.com/SIAT-SongYuxin/CT2USforKidneySeg.
This dataset is from the publication https://doi.org/10.1016/j.ultras.2022.106706.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def get_ct2us_kidney_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CT2US for Kidney Segmentation dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "slice", "slice")
    if os.path.exists(data_dir):
        return path

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name="siatsyx/ct2usforkidneyseg", download=download)
    zip_path = os.path.join(path, "ct2usforkidneyseg.zip")
    util.unzip(zip_path=zip_path, dst=path)

    return path


def get_ct2us_kidney_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the CT2US for Kidney Segmentation data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_ct2us_kidney_data(path=path, download=download)

    image_paths = natsorted(glob(os.path.join(data_dir, "slice", "slice", "*.png")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "mask", "mask", "*.png")))
    assert len(image_paths) > 0 and len(image_paths) == len(gt_paths)

    neu_gt_dir = os.path.join(data_dir, "mask", "preprocessed")
    os.makedirs(neu_gt_dir, exist_ok=True)

    neu_gt_paths = []
    for gt_path in tqdm(gt_paths, desc="Preprocessing labels"):
        neu_gt_path = os.path.join(neu_gt_dir, f"{Path(gt_path).stem}.tif")
        neu_gt_paths.append(neu_gt_path)
        if os.path.exists(neu_gt_path):
            continue

        gt = imageio.imread(gt_path)
        gt = (gt >= 128).astype("uint8")
        imageio.imwrite(neu_gt_path, gt, compression="zlib")

    return image_paths, neu_gt_paths


def get_ct2us_kidney_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CT2US for Kidney Segmentation dataset for kidney segmentation in synthetic ultrasound images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_ct2us_kidney_paths(path, download)

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


def get_ct2us_kidney_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CT2US for Kidney Segmentation dataloader for kidney segmentation in synthetic ultrasound images.

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
    dataset = get_ct2us_kidney_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
