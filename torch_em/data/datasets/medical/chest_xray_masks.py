"""The Chest Xray Masks and Labels dataset contains annotations for lung
segmentation in chest x-ray images.

The dataset combines the Montgomery County and Shenzhen chest x-ray collections
with manually drawn lung field masks and is located at
https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels.
The underlying data is released by the National Library of Medicine, National
Institutes of Health, Bethesda, MD, USA and Shenzhen No.3 People's Hospital,
Guangdong Medical College, Shenzhen, China. This dataset is from the publications
https://doi.org/10.1109/TMI.2013.2284099 and https://doi.org/10.1109/TMI.2013.2290491.
Please cite them if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


KAGGLE_DATASET_NAME = "nikhilpandey360/chest-xray-masks-and-labels"


def get_chest_xray_masks_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Chest Xray Masks and Labels dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Lung Segmentation")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name=KAGGLE_DATASET_NAME, download=download)
    zip_path = os.path.join(path, "chest-xray-masks-and-labels.zip")
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_chest_xray_masks_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Chest Xray Masks and Labels data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_chest_xray_masks_data(path=path, download=download)

    mask_paths = natsorted(glob(os.path.join(data_dir, "masks", "*.png")))

    # NOTE: Masks for the Shenzhen images have a '_mask' suffix (eg. 'CHNCXR_0001_0_mask.png'), while masks
    # for the Montgomery images share the same filename as the corresponding image (eg. 'MCUCXR_0001_0.png').
    image_paths = []
    for mask_path in mask_paths:
        image_id = os.path.basename(mask_path).replace("_mask.png", ".png")
        image_paths.append(os.path.join(data_dir, "CXR_png", image_id))

    assert len(image_paths) == len(mask_paths) and len(image_paths) > 0
    assert all(os.path.exists(p) for p in image_paths)

    return image_paths, mask_paths


def get_chest_xray_masks_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Chest Xray Masks and Labels dataset for lung segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_chest_xray_masks_paths(path, download)

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


def get_chest_xray_masks_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Chest Xray Masks and Labels dataloader for lung segmentation.

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
    dataset = get_chest_xray_masks_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
