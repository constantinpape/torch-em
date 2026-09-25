"""The MuscleUS dataset contains annotations for cross-sectional muscle segmentation in
musculoskeletal ultrasound images.

The dataset consists of 3,917 transverse ultrasound images of the biceps brachii ('BB'), tibialis
anterior ('TA') and gastrocnemius medialis ('GM') muscles, acquired on 1,283 subjects (both healthy
and with neuromuscular disease), with manual binary masks of the muscle cross-sectional area.

The dataset is located at https://doi.org/10.17632/3jykz7wz8d.1 and is distributed under the
CC BY 4.0 license.
The dataset is from the publication https://doi.org/10.1016/j.compbiomed.2021.104623.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util
from ..light_microscopy.neurips_cell_seg import to_rgb


URL = "https://data.mendeley.com/public-files/datasets/3jykz7wz8d/files/b1601a22-98a5-41ce-9fbb-54dd21834b55/file_downloaded"  # noqa
CHECKSUM = "81dbec46e9f3035831f97aa828735cd6ba04171a72902a9416e7847e75148a11"

MUSCLES = ["BB", "GM", "TA"]
"""The choice of muscles in the dataset: biceps brachii ('BB'), gastrocnemius medialis ('GM')
and tibialis anterior ('TA')."""

CATEGORIES = ["Healthy", "Pathological"]
"""The choice of subject categories in the dataset."""


def get_muscle_us_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the MuscleUS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "Polito-Radboud-DeepLearningUS")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "muscle_us.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_muscle_us_paths(
    path: Union[os.PathLike, str],
    muscle: Optional[Literal["BB", "GM", "TA"]] = None,
    category: Optional[Literal["Healthy", "Pathological"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the MuscleUS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        muscle: The choice of muscle. One of 'BB', 'GM', 'TA'. By default, loads images of all muscles.
        category: The choice of subject category. One of 'Healthy', 'Pathological'.
            By default, loads images of all categories.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if muscle is None:
        muscles = MUSCLES
    elif muscle in MUSCLES:
        muscles = [muscle]
    else:
        raise ValueError(f"'{muscle}' is not a valid muscle. Choose from {MUSCLES}.")

    if category is None:
        categories = CATEGORIES
    elif category in CATEGORIES:
        categories = [category]
    else:
        raise ValueError(f"'{category}' is not a valid category. Choose from {CATEGORIES}.")

    data_dir = get_muscle_us_data(path, download)

    image_paths, label_paths = [], []
    for m in muscles:
        for c in categories:
            image_dir = os.path.join(data_dir, m, c, "Images")
            label_dir = os.path.join(data_dir, m, c, "Masks")
            for image_path in natsorted(glob(os.path.join(image_dir, "*.png"))):
                label_path = os.path.join(label_dir, os.path.basename(image_path))
                if os.path.exists(label_path):
                    image_paths.append(image_path)
                    label_paths.append(label_path)

    return image_paths, label_paths


def get_muscle_us_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    muscle: Optional[Literal["BB", "GM", "TA"]] = None,
    category: Optional[Literal["Healthy", "Pathological"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MuscleUS dataset for muscle cross-sectional area segmentation in ultrasound images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        muscle: The choice of muscle. One of 'BB', 'GM', 'TA'. By default, loads images of all muscles.
        category: The choice of subject category. One of 'Healthy', 'Pathological'.
            By default, loads images of all categories.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_muscle_us_paths(path, muscle, category, download)

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


def get_muscle_us_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    muscle: Optional[Literal["BB", "GM", "TA"]] = None,
    category: Optional[Literal["Healthy", "Pathological"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MuscleUS dataloader for muscle cross-sectional area segmentation in ultrasound images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        muscle: The choice of muscle. One of 'BB', 'GM', 'TA'. By default, loads images of all muscles.
        category: The choice of subject category. One of 'Healthy', 'Pathological'.
            By default, loads images of all categories.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_muscle_us_dataset(path, patch_shape, muscle, category, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
