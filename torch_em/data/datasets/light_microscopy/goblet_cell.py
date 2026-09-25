"""The GCS dataset contains annotations for goblet cell instance segmentation in
microscopy images of the human conjunctiva.

The dataset consists of 24 images (2048x1536, RGB) with 65,108 manually annotated single cells. It also
provides a patched version of the same images with 1,152 patches (256x256) and 75,597 instances, where cells
that cross patch borders are counted in every patch that contains them. The masks store one integer id per cell,
which is used directly as the instance label (touching cells are not merged).

The data is located at https://doi.org/10.5281/zenodo.18517381 (latest version: https://zenodo.org/records/18642562),
released under a CC-BY-4.0 license. The archive also contains YOLO and SAM2 annotation formats, which are not used.

The official splits are used for the unpatched images and for the patched images split by source image ('img_level').
The second official patched split ('random') puts patches of the same source image into train and test and is
therefore not exposed.

This dataset is from the publication https://doi.org/10.1038/s41597-026-07309-w.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/api/records/18642562/files/GCSdataV1.1.zip/content"
CHECKSUM = "e0c59c5d6c281b3f793b934e605c16ca80d8d75f59da3f2579ca6da9c0c0a973"

SPLITS = ["train", "test", "all"]


def get_goblet_cell_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the GCS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the original images and masks.
    """
    data_dir = os.path.join(path, "GCSdataV1.1", "original_data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "GCSdataV1.1.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path, remove=False)

    assert os.path.exists(data_dir), f"The extraction of the archive did not create the expected folder in '{path}'."

    return data_dir


def get_goblet_cell_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test", "all"] = "all",
    patched: bool = False,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the GCS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. One of 'train', 'test' or 'all'.
        patched: Whether to use the patched images (256x256) instead of the full images (2048x1536).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose one of {SPLITS}.")

    data_dir = get_goblet_cell_data(path, download)

    version = "patched" if patched else "unpatched"
    if split == "all":
        split_dir = os.path.join(data_dir, "complete", version)
    else:
        split_dir = os.path.join(data_dir, "train_test_split", "patched_img_level" if patched else "unpatched", split)

    raw_paths = natsorted(glob(os.path.join(split_dir, "images", "*")))
    label_paths = [
        os.path.join(split_dir, "masks", f"{os.path.splitext(os.path.basename(p))[0]}.png") for p in raw_paths
    ]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_goblet_cell_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test", "all"] = "all",
    patched: bool = False,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the GCS dataset for goblet cell instance segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. One of 'train', 'test' or 'all'.
        patched: Whether to use the patched images (256x256) instead of the full images (2048x1536).
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_goblet_cell_paths(path, split, patched, download)

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


def get_goblet_cell_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test", "all"] = "all",
    patched: bool = False,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the GCS dataloader for goblet cell instance segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. One of 'train', 'test' or 'all'.
        patched: Whether to use the patched images (256x256) instead of the full images (2048x1536).
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_goblet_cell_dataset(path, patch_shape, split, patched, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
