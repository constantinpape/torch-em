"""The PALM dataset contains annotations for optic disc and lesion segmentation
in Fundus images.

The dataset is from the publication https://doi.org/10.1038/s41597-024-02911-2.
Please cite it if you use this dataset for your research.
"""

import os
import shutil
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://springernature.figshare.com/ndownloader/files/37786152"
CHECKSUM = "21cd568a00a50287370572ea81b50847085819bd2f732331ee9cdc6367e6cd1f"


def get_palm_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PALM data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "PALM")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "data.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    shutil.rmtree(os.path.join(path, "__MACOSX"))

    return data_dir


def _preprocess_labels(label_paths):
    neu_label_paths = [p.replace(".bmp", "_preprocessed.tif") for p in label_paths]
    for lpath, neu_lpath in zip(label_paths, neu_label_paths):
        if os.path.exists(neu_lpath):
            continue

        label = imageio.imread(lpath)
        imageio.imwrite(neu_lpath, (label == 0).astype(int), compression="zlib")

    return neu_label_paths


def get_palm_paths(
    path: Union[os.PathLike, str],
    split: Literal["Training", "Validation", "Testing"],
    label_choice: Literal["disc", "atrophy_lesion", "detachment_lesion"] = "disc",
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the PALM data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        label_choice: The choice of label masks.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_palm_data(path, download)

    assert split in ["Training", "Validation", "Testing"], f"'{split}' is not a valid split."

    if label_choice == "disc":
        ldir = "Disc Masks"
    elif label_choice == "atrophy_lesion":
        ldir = "Lesion Masks/Atrophy"
    elif label_choice == "detachment_lesion":
        ldir = "Lesion Masks/Detachment"
    else:
        raise ValueError(f"'{label_choice}' is not a valid choice of labels.")

    label_paths = natsorted(glob(os.path.join(data_dir, split, ldir, "*.bmp")))
    label_paths = _preprocess_labels(label_paths)

    raw_paths = [p.replace(ldir, "Images") for p in label_paths]
    raw_paths = [p.replace("_preprocessed.tif", ".jpg") for p in raw_paths]

    assert len(label_paths) == len(raw_paths)

    return raw_paths, label_paths


def get_palm_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["Training", "Validation", "Testing"],
    label_choice: Literal["disc", "atrophy_lesion", "detachment_lesion"] = "disc",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PALM dataset for disc and lesion segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        label_choice: The choice of label masks.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_palm_paths(path, split, label_choice, download)

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
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_palm_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["Training", "Validation", "Testing"],
    label_choice: Literal["disc", "atrophy_lesion", "detachment_lesion"] = "disc",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PALM dataloader for disc and lesion segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        label_choice: The choice of label masks.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_palm_dataset(path, patch_shape, split, label_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
