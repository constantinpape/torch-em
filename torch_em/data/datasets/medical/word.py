"""The WORD dataset contains annotations for 16 abdominal organs in CT scans.

The dataset consists of 150 abdominal CT volumes with an official split into 100 training, 20 validation
and 30 test volumes. The label ids are: 1: liver, 2: spleen, 3: kidney (left), 4: kidney (right), 5: stomach,
6: gallbladder, 7: esophagus, 8: pancreas, 9: duodenum, 10: colon, 11: intestine, 12: adrenal, 13: rectum,
14: bladder, 15: head of femur (left), 16: head of femur (right). See also `CLASS_IDS`.

The dataset is located at https://github.com/HiLab-git/WORD. It is distributed as a password protected zip archive
via Google Drive (the password 'word@uestc' is given in the repository).

NOTE: The Google Drive download is quota limited, so it intermittently fails with
'Too many users have viewed or downloaded this file recently'. Retrying later usually works. Otherwise download
'WORD-V0.1.0.zip' manually from https://drive.google.com/drive/folders/16qwlCxH7XtJD9MyPnAbmY4ATxu2mKu67
and place it in `path`.

This dataset is from the publication https://doi.org/10.1016/j.media.2022.102642.
Please cite it if you use this dataset in your research.
"""

import os
import zipfile
from glob import glob
from shutil import which
from subprocess import run
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "data": "https://drive.google.com/uc?id=19OWCXZGrimafREhXm8O8w2HBHZTfxEgU",
    "test_labels": "https://github.com/HiLab-git/WORD/raw/main/WORD_V0.1.0_labelsTs.zip",
}

CHECKSUMS = {
    "data": "1ef1e48f8b41d72d733c5cdd91e1238d9b345d88c0de0c902e356310901f3115",
    "test_labels": "0552b1e208a8a5345a1c12003d4f1d010621c6ef65bdd62bff85a32b2976a46d",
}

PASSWORD = "word@uestc"

CLASS_NAMES = [
    "liver", "spleen", "kidney_left", "kidney_right", "stomach", "gallbladder", "esophagus", "pancreas", "duodenum",
    "colon", "intestine", "adrenal", "rectum", "bladder", "head_of_femur_left", "head_of_femur_right",
]
"""The organs of the WORD dataset. The label id of an organ is its 1-based index."""

CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
"""Mapping from the organ name to its label id."""

SPLIT_DIRS = {"train": "Tr", "val": "Val", "test": "Ts"}


def _unzip_with_password(zip_path, dst, password):
    try:
        with zipfile.ZipFile(zip_path) as f:
            f.extractall(dst, pwd=password.encode())
    except (NotImplementedError, RuntimeError) as e:
        # The python zipfile module does not support AES encryption, so we fall back to the 7z CLI.
        if which("7z") is None:
            raise RuntimeError(
                f"Could not extract '{zip_path}' with the zipfile module ({e}). Please install the '7z' CLI "
                "('conda install -c conda-forge p7zip') or extract the archive manually."
            )
        run(["7z", "x", f"-o{dst}", f"-p{password}", "-y", zip_path], check=True)


def _find_data_dir(path):
    candidates = glob(os.path.join(path, "imagesTr")) + glob(os.path.join(path, "*", "imagesTr"))
    return os.path.dirname(candidates[0]) if candidates else None


def get_word_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the WORD dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = _find_data_dir(path)
    if data_dir is None:
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, "WORD-V0.1.0.zip")
        util.download_source_gdrive(path=zip_path, url=URLS["data"], download=download, checksum=CHECKSUMS["data"])
        if not os.path.exists(zip_path):
            raise RuntimeError(
                "The automatic download of the WORD dataset failed, most likely because the Google Drive download "
                "quota of the archive is exceeded. Please download 'WORD-V0.1.0.zip' manually from "
                f"https://drive.google.com/drive/folders/16qwlCxH7XtJD9MyPnAbmY4ATxu2mKu67 and place it at "
                f"'{zip_path}'."
            )
        _unzip_with_password(zip_path, path, PASSWORD)
        data_dir = _find_data_dir(path)
        if data_dir is None:
            raise RuntimeError(f"Could not find the 'imagesTr' folder of the WORD dataset in '{path}'.")

    # The test labels were released separately via the GitHub repository.
    if not os.path.exists(os.path.join(data_dir, "labelsTs")):
        zip_path = os.path.join(path, "WORD_V0.1.0_labelsTs.zip")
        util.download_source(
            path=zip_path, url=URLS["test_labels"], download=download, checksum=CHECKSUMS["test_labels"]
        )
        util.unzip(zip_path=zip_path, dst=data_dir, remove=False)

    return data_dir


def get_word_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the WORD data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLIT_DIRS:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLIT_DIRS.keys())}.")

    data_dir = get_word_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, f"images{SPLIT_DIRS[split]}", "*.nii.gz")))
    label_paths = [p.replace(f"images{SPLIT_DIRS[split]}", f"labels{SPLIT_DIRS[split]}") for p in raw_paths]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_word_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the WORD dataset for abdominal organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_word_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_word_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the WORD dataloader for abdominal organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_word_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
