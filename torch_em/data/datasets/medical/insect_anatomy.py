"""The InsectAnatomy dataset contains annotations for brain segmentation in micro-CT scans of ant heads.

It comprises 2d cross-sections (along the xy, xz and yz planes) of the head scans of 76 ant species with
semi-manually segmented brain masks, split into an official training (60%) and testing (40%) set.
The images and masks are stored as tif files. The label ids are: 0 = background, 1 = brain.
This is the 'InsectAnatomy' out-of-distribution set of the nnInteractive benchmark
(https://arxiv.org/abs/2503.08373), which reports 84 micro-CT volumes for it.

NOTE: The data is hosted on Dryad (https://doi.org/10.5061/dryad.qz612jmgv), which protects its downloads
with a browser check that automated downloads cannot pass. If the automatic download fails, please download
'training.zip' and 'testing.zip' from the Dryad page in your browser and place them at '<path>'.

NOTE: The layout inside the archives is not documented by the authors. The images and masks are matched by
their file names (see `MASK_KEYWORDS`); adjust `get_insect_anatomy_paths` if the layout differs. The archives
mix three naming conventions and also contain unpaired files, i.e. species for which only the images or only
the masks were deposited, so those are skipped: this yields 35422 pairs for 'train' and 7366 for 'test'.

NOTE: The masks are 8 bit images with antialiased borders, where the brain is 255 and the background is 0.
They are binarized to the label ids above by the dataset, so a custom 'label_transform' overrides this.

NOTE: The masks of the test split are inconsistent and it is therefore not recommended for training. In the
training split the brain is marked correctly (247 of 250 sampled masks cover the brighter brain tissue, with a
mean foreground of 22%), but in the test split most masks have an inverted polarity, i.e. they mark the
background instead of the brain: of 179 sampled masks that cover more than half of the image, only 6 cover the
brighter tissue. The deposit does not document this, and the polarity cannot be told apart from a correctly
annotated slice of a cropped scan, so the masks are passed through as they are stored and a warning is raised.

The dataset is located at https://doi.org/10.5061/dryad.qz612jmgv.

This dataset is from the publication https://doi.org/10.1002/ntls.20230010.
Please cite it if you use this dataset in your research.
"""

import os
import re
import zipfile
import warnings
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "train": "https://datadryad.org/downloads/file_stream/2625340",
    "test": "https://datadryad.org/downloads/file_stream/2625339",
}

ZIP_NAMES = {"train": "training.zip", "test": "testing.zip"}

LABEL_IDS = {"background": 0, "brain": 1}

MASK_KEYWORDS = ("mask", "label", "seg", "gt")

MANUAL_DOWNLOAD_MSG = (
    "The automatic download of '{}' from Dryad failed. Dryad protects its downloads with a browser check. "
    "Please download the file manually from https://doi.org/10.5061/dryad.qz612jmgv and place it at '{}'."
)


def _binarize_labels(labels):
    # The masks are 8 bit images with antialiased borders, where the brain is 255 and the background is 0.
    return (np.asarray(labels) > 127).astype("uint8")


def get_insect_anatomy_data(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> str:
    """Download the InsectAnatomy dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    if split not in URLS:
        raise ValueError(f"'{split}' is not a valid split.")

    data_dir = os.path.join(path, split)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, ZIP_NAMES[split])
    if not os.path.exists(zip_path):
        if not download:
            raise RuntimeError(f"Cannot find the data at {zip_path}, but download was set to False.")

        try:
            util.download_source(path=zip_path, url=URLS[split], download=download)
        except Exception:
            raise RuntimeError(MANUAL_DOWNLOAD_MSG.format(ZIP_NAMES[split], path))

    if not zipfile.is_zipfile(zip_path):
        os.remove(zip_path)
        raise RuntimeError(MANUAL_DOWNLOAD_MSG.format(ZIP_NAMES[split], path))

    util.unzip(zip_path=zip_path, dst=data_dir)
    return data_dir


def get_insect_anatomy_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the InsectAnatomy data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_insect_anatomy_data(path, split, download)

    if split == "test":
        warnings.warn(
            "The masks of the InsectAnatomy test split are inconsistent: most of them mark the background "
            "instead of the brain. See 'torch_em.data.datasets.medical.insect_anatomy' for details."
        )

    # The images and their masks are stored as tif files, the masks are marked by 'mask' in the file name.
    # NOTE: The archives use three different naming conventions, so the name of the image is derived from the
    # name of the mask by removing the first mask marker: '<name>.tif' pairs with '<name>_mask.tif', and
    # '<species><slice>.tif' pairs with both '<species>__mask<slice>.tif' and '<species>_mask<slice>.tif'.
    all_paths = natsorted(glob(os.path.join(data_dir, "**", "*.tif*"), recursive=True))

    # The archives contain the sidecar files that macOS adds when zipping, which are not image data.
    all_paths = [p for p in all_paths if "__MACOSX" not in p and not os.path.basename(p).startswith("._")]

    mask_paths, images_by_name = [], {}
    for filepath in all_paths:
        name = os.path.splitext(os.path.basename(filepath))[0]
        if any(kw in name.lower() for kw in MASK_KEYWORDS):
            mask_paths.append(filepath)
        else:
            images_by_name[name] = filepath

    raw_paths, label_paths = [], []
    for filepath in mask_paths:
        name = os.path.splitext(os.path.basename(filepath))[0]
        image_name = re.sub(r"_{0,2}mask", "", name, count=1)
        if image_name in images_by_name:
            raw_paths.append(images_by_name[image_name])
            label_paths.append(filepath)

    if len(raw_paths) == 0:
        raise RuntimeError(
            f"Found {len(images_by_name)} images and {len(mask_paths)} masks in '{data_dir}', but could not match "
            "any of them. The images and masks are matched by their file names, see 'MASK_KEYWORDS'."
        )

    return raw_paths, label_paths


def get_insect_anatomy_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the InsectAnatomy dataset for ant brain segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_insect_anatomy_paths(path, split, download)

    # The masks are stored as 8 bit images with antialiased borders, so they are binarized to the documented ids.
    kwargs = util.update_kwargs(kwargs, "label_transform", _binarize_labels)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_insect_anatomy_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the InsectAnatomy dataloader for ant brain segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
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
    dataset = get_insect_anatomy_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
