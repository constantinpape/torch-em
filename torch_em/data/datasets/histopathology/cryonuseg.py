"""The CryoNuSeg dataset contains annotations for nucleus segmentation
in cryosectioned H&E stained histological images of 10 different organs.

This dataset is from the publication https://doi.org/10.1016/j.compbiomed.2021.104349.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def get_cryonuseg_data(path: Union[os.PathLike, str], download: bool = False):
    """Download the CryoNuSeg dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.
    """
    if os.path.exists(os.path.join(path, r"tissue images")):
        return

    os.makedirs(path, exist_ok=True)
    util.download_source_kaggle(
        path=path, dataset_name="ipateam/segmentation-of-nuclei-in-cryosectioned-he-images", download=download
    )

    zip_path = os.path.join(path, "segmentation-of-nuclei-in-cryosectioned-he-images.zip")
    util.unzip(zip_path=zip_path, dst=path)


def get_cryonuseg_paths(
    path: Union[os.PathLike, str], rater_choice: Literal["b1", "b2", "b3"] = "b1", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the CryoNuSeg data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        rater: The choice of annotator.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the image data.
        List of filepaths to the label data.
    """
    get_cryonuseg_data(path, download)

    if rater_choice == "b1":
        label_dir = r"Annotator 1 (biologist)/"
    elif rater_choice == "b2":
        label_dir = r"Annotator 1 (biologist second round of manual marks up)/" * 2
    elif rater_choice == "b3":
        label_dir = r"Annotator 2 (bioinformatician)/" * 2
    else:
        raise ValueError(f"'{rater_choice}' is not a valid rater choice.")

    # Point to the instance labels folder
    label_dir += r"label masks modify"

    label_paths = natsorted(glob(os.path.join(path, label_dir, "*.tif")))
    raw_paths = natsorted(glob(os.path.join(path, r"tissue images", "*.tif")))

    return raw_paths, label_paths


def get_cryonuseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    rater: Literal["b1", "b2", "b3"] = "b1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CryoNuSeg dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        rater: The choice of annotator.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cryonuseg_paths(path, rater, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_cryonuseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    rater: Literal["b1", "b2", "b3"] = "b1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CryoNuSeg dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        rater: The choice of annotator.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cryonuseg_dataset(path, patch_shape, rater, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
