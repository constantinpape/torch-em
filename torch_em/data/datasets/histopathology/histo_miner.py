"""The Histo-Miner dataset contains annotations for nucleus instance and semantic segmentation
(NucSeg) and for tumor region segmentation (TumSeg) in H&E stained cutaneous squamous cell
carcinoma (cSCC) histopathology images.

NOTE: The public deposit only exposes train and validation splits; the held-out TumSeg test
split is released in a separate Zenodo record and is not covered by this loader.

This dataset is located at https://zenodo.org/records/15973142.
This dataset is from the publication https://doi.org/10.1371/journal.pcbi.1013907.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Literal, Tuple, List

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "nuclei": "https://zenodo.org/api/records/15973142/files/NucSeg_OriginalFormat.zip/content",
    "tumor": "https://zenodo.org/api/records/15973142/files/TumSeg.zip/content",
}

CHECKSUMS = {
    "nuclei": "6316b027ef50ce874e3f147f20a069f6c5ad9af5688d919c1ecf836301e6eccb",
    "tumor": "8091907d84ef75cfa7cf5deff113f6d9d2a76f9269c29f095703165c4f91f682",
}

DATA_DIRNAMES = {"nuclei": "NucSeg_OriginalFormat", "tumor": "TumSeg"}


def get_histo_miner_data(
    path: Union[os.PathLike, str], task: Literal["nuclei", "tumor"], download: bool = False
) -> str:
    """Download the Histo-Miner data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        task: The choice of task, either nucleus segmentation ('nuclei') or tumor region segmentation ('tumor').
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the dataset is downloaded and stored for further preprocessing.
    """
    data_dir = os.path.join(path, DATA_DIRNAMES[task])
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"histo_miner_{task}.zip")
    util.download_source(path=zip_path, url=URLS[task], download=download, checksum=CHECKSUMS[task])
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _convert_nuclei_npy_to_tif(data_dir, split_dir, label_choice):
    label_dirname = "InstanceMaps" if label_choice == "instances" else "ClassMaps"
    raw_dir = os.path.join(data_dir, split_dir, "RawImages")
    label_dir = os.path.join(data_dir, split_dir, label_dirname)

    converted_raw_dir = os.path.join(data_dir, split_dir, "RawImages_tif")
    converted_label_dir = os.path.join(data_dir, split_dir, f"{label_dirname}_tif")
    os.makedirs(converted_raw_dir, exist_ok=True)
    os.makedirs(converted_label_dir, exist_ok=True)

    raw_paths, label_paths = [], []
    for raw_path in natsorted(glob(os.path.join(raw_dir, "*.npy"))):
        fname = os.path.basename(raw_path).replace(".npy", ".tif")

        out_raw_path = os.path.join(converted_raw_dir, fname)
        if not os.path.exists(out_raw_path):
            imageio.imwrite(out_raw_path, np.load(raw_path))
        raw_paths.append(out_raw_path)

        label_path = os.path.join(label_dir, os.path.basename(raw_path))
        out_label_path = os.path.join(converted_label_dir, fname)
        if not os.path.exists(out_label_path):
            imageio.imwrite(out_label_path, np.load(label_path).astype("int32"))
        label_paths.append(out_label_path)

    return raw_paths, label_paths


def get_histo_miner_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val"],
    task: Literal["nuclei", "tumor"] = "nuclei",
    label_choice: Literal["instances", "semantic"] = "instances",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Histo-Miner data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        task: The choice of task, either nucleus segmentation ('nuclei') or tumor region segmentation ('tumor').
        label_choice: The choice of label representation for the 'nuclei' task, either instance or semantic labels.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the image data.
        List of filepaths to the label data.
    """
    data_dir = get_histo_miner_data(path, task, download)
    split_dir = "Train" if split == "train" else "Val"

    if task == "nuclei":
        raw_paths, label_paths = _convert_nuclei_npy_to_tif(data_dir, split_dir, label_choice)
    else:
        raw_paths = natsorted(glob(os.path.join(data_dir, split_dir, "images", "*.tif")))
        label_paths = natsorted(glob(os.path.join(data_dir, split_dir, "annotations", "*.png")))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    return raw_paths, label_paths


def get_histo_miner_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"],
    task: Literal["nuclei", "tumor"] = "nuclei",
    label_choice: Literal["instances", "semantic"] = "instances",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Histo-Miner dataset for nucleus or tumor region segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        task: The choice of task, either nucleus segmentation ('nuclei') or tumor region segmentation ('tumor').
        label_choice: The choice of label representation for the 'nuclei' task, either instance or semantic labels.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_histo_miner_paths(path, split, task, label_choice, download)

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
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_histo_miner_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"],
    task: Literal["nuclei", "tumor"] = "nuclei",
    label_choice: Literal["instances", "semantic"] = "instances",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Histo-Miner dataloader for nucleus or tumor region segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        task: The choice of task, either nucleus segmentation ('nuclei') or tumor region segmentation ('tumor').
        label_choice: The choice of label representation for the 'nuclei' task, either instance or semantic labels.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_histo_miner_dataset(
        path, patch_shape, split, task, label_choice, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
