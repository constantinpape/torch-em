"""The LongCIU dataset contains annotations for ground glass opacity and consolidation
in chest CT scans of Long COVID patients.

This dataset is located at https://doi.org/10.25820/data.007301, and is released
under the Open Data Commons Attribution License (ODC-By) v1.0.
The dataset is from the publication https://doi.org/10.1038/s41597-025-04709-2.
Please cite it if you use this dataset for your research.
"""

import os
import json
from glob import glob
from natsort import natsorted
from typing import Tuple, Union, Literal, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://iro.uiowa.edu/view/fileRedirect?instCode=01IOWA_INST&download=true&filePid={}"

FILE_PIDS = {
    "img": "13931395260002771",
    "staple": "13931395270002771",
    "1": "13931395300002771",
    "2": "13931395290002771",
    "3": "13931395280002771",
    "splits": "13931395240002771",
}

FNAMES = {
    "img": "longciu_img.nii.gz",
    "staple": "longciu_STAPLE_tgt.nii.gz",
    "1": "longciu_1_tgt.nii.gz",
    "2": "longciu_2_tgt.nii.gz",
    "3": "longciu_3_tgt.nii.gz",
    "splits": "longciu_splits.json",
}

CHECKSUMS = {
    "img": "af54aff713e7054d204b5e5e2aa0598c856006229901eea791ece8ae2efe8ab8",
    "staple": "6f0d64315c5010b296285f81a54adec0c6eae5fc643c4e21f4546129d743ea13",
    "1": "f9cd0e605f6fcd701a64a262ba15d43a46fe2f1b62a4045e83195e4962f022ed",
    "2": "4e54ec5add75b2ee48f34be6bd25c74f23d550a96430ff10c04e711383f9647f",
    "3": "aee04feddb18cbbe717418ce8b6838b58fc415f0b5b13b97b1c682faed1bb7cd",
    "splits": "7f4d62f33f68a16cd2df10cc320e087d43802970de63e6e459885f5c7e5604bf",
}


def _preprocess_data(path):
    import h5py
    import nibabel as nib

    def _load(name):
        vol = nib.load(os.path.join(path, FNAMES[name])).get_fdata()
        return vol.transpose(2, 0, 1)

    raw = _load("img").astype("float32")
    labels = {rater: np.rint(_load(rater)).astype("uint8") for rater in ["staple", "1", "2", "3"]}

    with open(os.path.join(path, FNAMES["splits"])) as f:
        splits = json.load(f)

    data_dir = os.path.join(path, "data")
    os.makedirs(data_dir, exist_ok=True)

    for split, ids in splits.items():
        ids = sorted(ids)
        with h5py.File(os.path.join(data_dir, f"longciu_{split}.h5"), "w") as f:
            f.create_dataset("raw", data=raw[ids], compression="gzip")
            for rater, vol in labels.items():
                f.create_dataset(f"labels/{rater}", data=vol[ids], compression="gzip")

    for name in ["img", "staple", "1", "2", "3"]:
        os.remove(os.path.join(path, FNAMES[name]))


def get_longciu_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the LongCIU dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    for name, pid in FILE_PIDS.items():
        fpath = os.path.join(path, FNAMES[name])
        util.download_source(path=fpath, url=BASE_URL.format(pid), download=download, checksum=CHECKSUMS[name])

    _preprocess_data(path)

    return data_dir


def get_longciu_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False
) -> List[str]:
    """Get paths to the LongCIU data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the volumetric data.
    """
    data_dir = get_longciu_data(path, download)

    if split not in ("train", "val", "test"):
        raise ValueError(f"'{split}' is not a valid split.")

    volume_paths = natsorted(glob(os.path.join(data_dir, f"longciu_{split}.h5")))
    return volume_paths


def get_longciu_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    annotator: Literal["staple", "1", "2", "3"] = "staple",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LongCIU dataset for ground glass opacity and consolidation segmentation in chest CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        annotator: The choice of annotator providing the labels. Use 'staple' for the consensus segmentation.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_longciu_paths(path, split, download)

    if annotator not in ("staple", "1", "2", "3"):
        raise ValueError(f"'{annotator}' is not a valid choice of annotator.")

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=f"labels/{annotator}",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs,
    )


def get_longciu_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    annotator: Literal["staple", "1", "2", "3"] = "staple",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LongCIU dataloader for ground glass opacity and consolidation segmentation in chest CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        annotator: The choice of annotator providing the labels. Use 'staple' for the consensus segmentation.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_longciu_dataset(path, patch_shape, split, annotator, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
