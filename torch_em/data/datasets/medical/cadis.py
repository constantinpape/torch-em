"""The CaDIS dataset contains annotations for anatomy and surgical instrument segmentation
in RGB video frames of cataract surgery.

The dataset consists of 4670 frames sampled from 25 videos of the training set of the CATARACTS
challenge (https://cataracts.grand-challenge.org). Each frame is densely labeled at the pixel level
into up to 36 semantic classes: 4 anatomical structures (pupil, iris, cornea, skin), 29 surgical
instrument classes and 3 other object classes (surgical tape, hand, eye retractors). The labels are
stored as single-channel images, where the pixel value is the class index.

NOTE: The official train/val/test split assigns videos 5, 7 and 16 to validation and videos 2, 12
and 22 to test, with the remaining videos used for training (see `VAL_VIDEOS` and `TEST_VIDEOS`).

NOTE: The dataset is hosted on the CATARACTS Grand Challenge platform at
https://cataracts.grand-challenge.org/CaDIS/ and requires registration, so it cannot be downloaded
automatically. As of writing this, the direct download link on that page is not yet public
("Uploading - Link coming up soon"), so the data must currently be requested from the challenge
organizers. To obtain it:
- Visit https://cataracts.grand-challenge.org/CaDIS/ and register / log in via the challenge portal.
- Follow the instructions on the 'Data' / 'Download' pages of the challenge to request access to CaDIS.
- Once you have the data, place it such that each video folder is located at
  '<path>/CADIS/segmentation/Video<NN>/Images/*' and '<path>/CADIS/segmentation/Video<NN>/Labels/*',
  matching the structure of the original release.

This dataset is from the publication https://doi.org/10.1016/j.media.2021.102053.
Please cite it if you use this dataset in your research.
"""

import os
import re
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


VAL_VIDEOS = [5, 7, 16]
TEST_VIDEOS = [2, 12, 22]

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg")


def _video_number(video_dir):
    match = re.search(r"(\d+)", os.path.basename(video_dir))
    if match is None:
        raise ValueError(f"Could not parse a video number from '{video_dir}'.")
    return int(match.group(1))


def get_cadis_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Obtain the CaDIS dataset.

    Args:
        path: Filepath to a folder where the manually downloaded data is stored.
        download: Whether to download the data if it is not present. The data cannot be downloaded
            automatically, so this raises if the data has not been downloaded manually.

    Returns:
        Filepath to the folder with the per-video 'Images' and 'Labels' subfolders.
    """
    data_dir = os.path.join(path, "CADIS", "segmentation")
    if os.path.exists(data_dir) and glob(os.path.join(data_dir, "Video*")):
        return data_dir

    msg = "'torch_em' cannot download this dataset, because CaDIS is distributed via the CATARACTS Grand "
    msg += "Challenge and requires registration. Please visit 'https://cataracts.grand-challenge.org/CaDIS/', "
    msg += "register / log in and request access to the data via the challenge's 'Data' page, then place it "
    msg += f"such that video folders are located at '{data_dir}/Video<NN>/Images' and '.../Video<NN>/Labels'."
    raise NotImplementedError(msg)


def get_cadis_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the CaDIS data.

    Args:
        path: Filepath to a folder where the manually downloaded data is stored.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_cadis_data(path, download)

    video_dirs = natsorted(glob(os.path.join(data_dir, "Video*")))
    assert video_dirs, f"Did not find any 'Video*' folders in '{data_dir}'."

    if split == "val":
        video_dirs = [v for v in video_dirs if _video_number(v) in VAL_VIDEOS]
    elif split == "test":
        video_dirs = [v for v in video_dirs if _video_number(v) in TEST_VIDEOS]
    elif split == "train":
        video_dirs = [v for v in video_dirs if _video_number(v) not in VAL_VIDEOS + TEST_VIDEOS]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    image_paths, label_paths = [], []
    for video_dir in video_dirs:
        vimage_paths = []
        for ext in IMAGE_EXTENSIONS:
            vimage_paths.extend(glob(os.path.join(video_dir, "Images", ext)))
        vimage_paths = natsorted(vimage_paths)

        vlabel_paths = natsorted(glob(os.path.join(video_dir, "Labels", "*.png")))

        assert vimage_paths and len(vimage_paths) == len(vlabel_paths), \
            f"The images and labels for '{video_dir}' do not match."

        image_paths.extend(vimage_paths)
        label_paths.extend(vlabel_paths)

    return image_paths, label_paths


def get_cadis_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CaDIS dataset for anatomy and surgical instrument segmentation.

    Args:
        path: Filepath to a folder where the manually downloaded data is stored.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_cadis_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_cadis_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CaDIS dataloader for anatomy and surgical instrument segmentation.

    Args:
        path: Filepath to a folder where the manually downloaded data is stored.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cadis_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
