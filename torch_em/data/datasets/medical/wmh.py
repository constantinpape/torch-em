"""The WMH dataset contains annotations for white matter hyperintensity segmentation
in brain MRI (FLAIR and T1) from the MICCAI 2017 WMH Segmentation Challenge.

The data comprises 60 training and 110 test subjects, acquired at three sites (Utrecht, Singapore and Amsterdam)
on five different scanners. Each subject has a FLAIR and a T1 scan (the T1 has been registered to the FLAIR space
and both have been bias field corrected in the provided 'pre' folder) and a manual annotation in FLAIR space.

NOTE: The label legend is as follows:
- background: 0, white matter hyperintensity: 1, other pathology: 2

The dataset is located at https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/AECRSD.
The challenge website is https://wmh.isi.uu.nl/.

This dataset is from the publication https://doi.org/10.1109/TMI.2019.2905770.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List, Optional

import requests

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


DOI = "doi:10.34894/AECRSD"
API_URL = "https://dataverse.nl/api"
URL = f"{API_URL}/datasets/:persistentId/?persistentId={DOI}"

# NOTE: The dataset is downloaded file-by-file via the dataverse API (the full-dataset zip times out),
# so there is no single archive checksum.
CHECKSUM = None

SPLIT_DIRS = {"train": "training", "test": "test"}
N_SUBJECTS = {"train": 60, "test": 110}
FILENAMES = ["FLAIR.nii.gz", "T1.nii.gz", "wmh.nii.gz"]


def _get_wmh_file_list(split):
    response = requests.get(URL)
    response.raise_for_status()
    files = response.json()["data"]["latestVersion"]["files"]

    to_download = []
    for f in files:
        parts = f.get("directoryLabel", "").split("/")
        fname = f["dataFile"]["filename"]
        if parts[0] != SPLIT_DIRS[split] or fname not in FILENAMES:
            continue

        # The FLAIR and T1 scans are used from the preprocessed 'pre' folder (bias field corrected, T1 in FLAIR space).
        # The annotation 'wmh.nii.gz' is located directly in the subject folder.
        if parts[-1] == "pre":
            parts = parts[:-1]
        elif fname != "wmh.nii.gz":
            continue

        # The Amsterdam Philips scanner folder is named 'Philips_VU .PETMR_01.', we shorten it to 'Philips_VU'.
        parts = [p.split(" ")[0] for p in parts]
        to_download.append((os.path.join(*parts, fname), f["dataFile"]["id"]))

    return to_download


def get_wmh_data(path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False) -> str:
    """Download the WMH dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data for the split is stored.
    """
    if split not in SPLIT_DIRS:
        raise ValueError(f"'{split}' is not a valid split. Please choose from {list(SPLIT_DIRS.keys())}.")

    data_dir = os.path.join(path, SPLIT_DIRS[split])
    label_paths = glob(os.path.join(data_dir, "**", "wmh.nii.gz"), recursive=True)
    if len(label_paths) == N_SUBJECTS[split]:
        return data_dir

    if not download:
        raise RuntimeError(f"Cannot find the data at {data_dir}, but download was set to False.")

    for rel_path, file_id in _get_wmh_file_list(split):
        fpath = os.path.join(path, rel_path)
        os.makedirs(os.path.split(fpath)[0], exist_ok=True)
        util.download_source(path=fpath, url=f"{API_URL}/access/datafile/{file_id}", download=download, checksum=None)

    return data_dir


def get_wmh_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    modality: Optional[Literal["FLAIR", "T1"]] = None,
    site: Optional[Literal["Utrecht", "Singapore", "Amsterdam"]] = None,
    download: bool = False,
) -> Tuple[List[Union[str, Tuple[str, str]]], List[str]]:
    """Get paths to the WMH data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'test'.
        modality: The choice of modality. Either 'FLAIR' or 'T1'. By default, both are returned as channels.
        site: The acquisition site. One of 'Utrecht', 'Singapore' or 'Amsterdam'. By default, all sites are returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_wmh_data(path, split, download)

    site_dir = "*" if site is None else site
    label_paths = natsorted(glob(os.path.join(data_dir, site_dir, "**", "wmh.nii.gz"), recursive=True))
    if len(label_paths) == 0:
        raise ValueError(f"Could not find any data for split '{split}' and site '{site}'.")

    flair_paths = [p.replace("wmh.nii.gz", "FLAIR.nii.gz") for p in label_paths]
    t1_paths = [p.replace("wmh.nii.gz", "T1.nii.gz") for p in label_paths]
    assert all(os.path.exists(p) for p in flair_paths + t1_paths)

    if modality is None:
        raw_paths = [(fp, tp) for fp, tp in zip(flair_paths, t1_paths)]
    elif modality == "FLAIR":
        raw_paths = flair_paths
    elif modality == "T1":
        raw_paths = t1_paths
    else:
        raise ValueError(f"'{modality}' is not a valid modality. Please choose from 'FLAIR' or 'T1'.")

    return raw_paths, label_paths


def get_wmh_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "test"],
    modality: Optional[Literal["FLAIR", "T1"]] = None,
    site: Optional[Literal["Utrecht", "Singapore", "Amsterdam"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the WMH dataset for white matter hyperintensity segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        modality: The choice of modality. Either 'FLAIR' or 'T1'. By default, both are used as channels.
        site: The acquisition site. One of 'Utrecht', 'Singapore' or 'Amsterdam'. By default, all sites are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_wmh_paths(path, split, modality, site, download)

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
        with_channels=modality is None,
        **kwargs
    )


def get_wmh_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "test"],
    modality: Optional[Literal["FLAIR", "T1"]] = None,
    site: Optional[Literal["Utrecht", "Singapore", "Amsterdam"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the WMH dataloader for white matter hyperintensity segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'.
        modality: The choice of modality. Either 'FLAIR' or 'T1'. By default, both are used as channels.
        site: The acquisition site. One of 'Utrecht', 'Singapore' or 'Amsterdam'. By default, all sites are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_wmh_dataset(path, patch_shape, split, modality, site, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
