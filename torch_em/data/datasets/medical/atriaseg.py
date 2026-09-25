"""The AtriaSeg dataset contains annotations for left atrium segmentation in
late gadolinium enhanced (LGE) cardiac MRI of patients with atrial fibrillation.

The data was curated for the 2018 Atrial Segmentation Challenge
(https://www.cardiacatlas.org/atriaseg2018-challenge/), which was held together with the STACOM workshop at
MICCAI 2018. The release consists of 154 3D LGE-MRI of 154 patients, split into the 100 studies of the
official training set and the 54 studies of the official test set, which is selected with the 'split'
argument. All studies come with a segmentation of the left atrial cavity ('laendo.nrrd') and of the left
atrial wall ('lawall.nrrd'), which are merged into one label volume with the ids described in `LABEL_IDS`:
1 = left atrium cavity, 2 = left atrium wall. The two structures are disjoint. The challenge itself only
evaluated the left atrium cavity.

The LGE-MRI are stored as NRRD volumes with the slice axis last, so they are converted to hdf5 volumes with
the slice axis first (the keys are 'raw' and 'labels') by this module. All volumes have 88 slices.

NOTE: This requires the pynrrd python package.

The data is located at https://www.cardiacatlas.org/atriaseg2018-challenge/atria-seg-data/.

This dataset is from the publication https://doi.org/10.1016/j.media.2020.101832.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://www.dropbox.com/scl/fi/nero2nlaocdcdfhzwu5h0/2018_UTAH_MICCAI.zip?rlkey=vkkfrkc2l6x1e61jqyutb35qn&dl=1"  # noqa
CHECKSUM = "bee5ee5bd19a1caa1a375e147e56e7e691a4bc64e3873dc672d9d2b963a8f5e0"

LABEL_IDS = {"background": 0, "la_cavity": 1, "la_wall": 2}

SPLITS = {"train": "Training Set", "test": "Testing Set"}

N_VOLUMES = {"train": 100, "test": 54}


def _preprocess_inputs(split_dir, preprocessed_dir):
    import h5py
    import nrrd

    case_dirs = [p for p in natsorted(glob(os.path.join(split_dir, "*"))) if os.path.isdir(p)]
    os.makedirs(preprocessed_dir, exist_ok=True)

    for case_dir in tqdm(case_dirs, desc=f"Preprocessing the AtriaSeg cases of '{os.path.basename(split_dir)}'"):
        case_id = os.path.basename(case_dir)
        volume_path = os.path.join(preprocessed_dir, f"{case_id}.h5")
        if os.path.exists(volume_path):
            continue

        # The transpose maps the NRRD axis order (X, Y, Z) to the (Z, Y, X) order used for the volumes.
        raw = nrrd.read(os.path.join(case_dir, "lgemri.nrrd"))[0].T
        cavity = nrrd.read(os.path.join(case_dir, "laendo.nrrd"))[0].T
        wall = nrrd.read(os.path.join(case_dir, "lawall.nrrd"))[0].T

        # The masks are stored with the foreground value 255, which is mapped to the semantic label ids here.
        labels = np.zeros(raw.shape, dtype="uint8")
        labels[cavity > 0] = LABEL_IDS["la_cavity"]
        labels[wall > 0] = LABEL_IDS["la_wall"]

        # The file is written to a temporary path first, so that an interrupted run leaves no corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)


def get_atriaseg_data(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> str:
    """Download the AtriaSeg dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Please choose one of {list(SPLITS.keys())}.")

    preprocessed_dir = os.path.join(path, "preprocessed", split)
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == N_VOLUMES[split]:
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    split_dir = os.path.join(path, SPLITS[split])
    if not os.path.exists(split_dir):
        zip_path = os.path.join(path, "2018_UTAH_MICCAI.zip")
        util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip(zip_path=zip_path, dst=path)

    _preprocess_inputs(split_dir, preprocessed_dir)
    return preprocessed_dir


def get_atriaseg_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> List[str]:
    """Get paths to the AtriaSeg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_atriaseg_data(path, split, download)
    return natsorted(glob(os.path.join(data_dir, "*.h5")))


def get_atriaseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AtriaSeg dataset for left atrium segmentation.

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
    volume_paths = get_atriaseg_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_atriaseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the AtriaSeg dataloader for left atrium segmentation.

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
    dataset = get_atriaseg_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
