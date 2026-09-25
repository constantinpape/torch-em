"""The ParticleSeg3D dataset contains annotations for individual particle segmentation in micro CT images
of mineral, ore, slag and recycling samples.

The data consists of 54 annotated micro CT patches (41 for training and 13 for testing) that were extracted from
28 scanned samples of eight different materials. Each patch is paired with an instance segmentation in which every
individual particle carries its own id (0 is the background), so the label ids are instance ids and not semantic ids.
The 'split' argument exposes the official train / test split of the publication.

The data is hosted in a public share of the DESY sync-and-share service. This module downloads the 'Patches' folder
of that share (ca. 2.5 GB), which contains the annotated image patches ('images') and the corresponding instance
segmentations ('instance_seg') as nifti files. The 'Samples' folder of the share holds the full micro CT scans
(ca. 70 GB) without annotations and is therefore not downloaded here.

The dataset is located at https://syncandshare.desy.de/index.php/s/wjiDQ49KangiPj5 and the code of the publication
at https://github.com/MIC-DKFZ/ParticleSeg3D.

This dataset is from the publication https://doi.org/10.1016/j.powtec.2023.119286.
Please cite it if you use this dataset in your research.
"""

import os
import re
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from urllib.parse import quote, unquote
from typing import Union, Tuple, List, Literal

import requests

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


SHARE_TOKEN = "wjiDQ49KangiPj5"

URL = f"https://syncandshare.desy.de/index.php/s/{SHARE_TOKEN}"

WEBDAV_URL = "https://syncandshare.desy.de/public.php/webdav"

# The files are downloaded individually from the public share, so there is no checksum for a single archive.
CHECKSUM = None

# The number of annotated patches per split.
N_PATCHES = {"train": 41, "test": 13}

FOLDERS = ["images", "instance_seg"]


def _list_share_folder(folder):
    """List the file names in a folder of the public share via the WebDAV endpoint of Nextcloud.

    Public shares are accessed by using the share token as the user name and an empty password.
    """
    response = requests.request(
        "PROPFIND", f"{WEBDAV_URL}/{quote(folder)}/", headers={"Depth": "1"}, auth=(SHARE_TOKEN, "")
    )
    response.raise_for_status()

    fnames = []
    for href in re.findall(r"<d:href>(.*?)</d:href>", response.text):
        fname = unquote(href).rstrip("/").split("/")[-1]
        if fname.endswith(".nii.gz"):
            fnames.append(fname)
    return natsorted(fnames)


def _download_share_folder(folder, dst, download):
    """Download all nifti files of a folder of the public share into `dst`."""
    if os.path.exists(dst):
        return

    if not download:
        raise RuntimeError(f"Cannot find the data at {dst}, but download was set to False.")

    tmp_dir = f"{dst}.tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    fnames = _list_share_folder(folder)
    for fname in tqdm(fnames, desc=f"Download {len(fnames)} files from '{folder}'"):
        out_path = os.path.join(tmp_dir, fname)
        if os.path.exists(out_path):
            continue
        url = f"{URL}/download?path={quote('/' + folder)}&files={quote(fname)}"
        util.download_source(path=out_path, url=url, download=download, checksum=CHECKSUM)

    os.rename(tmp_dir, dst)


def get_particleseg3d_data(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> str:
    """Download the ParticleSeg3D dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    if split not in N_PATCHES:
        raise ValueError(f"'{split}' is not a valid split. Please choose one of {list(N_PATCHES.keys())}.")

    data_dir = os.path.join(path, "Patches", split)
    os.makedirs(path, exist_ok=True)
    for folder in FOLDERS:
        _download_share_folder(f"Patches/{split}/{folder}", os.path.join(data_dir, folder), download)

    return data_dir


def get_particleseg3d_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the ParticleSeg3D data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_particleseg3d_data(path, split, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "images", "*.nii.gz")))
    label_paths = [p.replace(os.sep + "images" + os.sep, os.sep + "instance_seg" + os.sep) for p in raw_paths]
    assert len(raw_paths) == N_PATCHES[split] and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_particleseg3d_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ParticleSeg3D dataset for particle instance segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_particleseg3d_paths(path, split, download)

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


def get_particleseg3d_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ParticleSeg3D dataloader for particle instance segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_particleseg3d_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
