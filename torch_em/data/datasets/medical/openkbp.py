"""The OpenKBP dataset contains annotations for organ-at-risk (OAR) and planning target volume (PTV)
segmentation in radiotherapy planning CT scans of head-and-neck cancer patients.

The data was curated for the OpenKBP Grand Challenge (https://github.com/ababier/open-kbp), which was held
at AAPM 2020 and targets models for knowledge-based dose prediction. The public release consists of 340
patients treated with intensity modulated radiation therapy, split into the official 200 training, 40
validation and 100 test studies (selected with the 'split' argument). Each patient has a 128x128x128 CT
volume together with up to 10 structure masks: 7 OARs ('Brainstem', 'SpinalCord', 'RightParotid',
'LeftParotid', 'Esophagus', 'Larynx', 'Mandible') and 3 PTVs ('PTV56', 'PTV63', 'PTV70'), see `LABEL_IDS`.
Not every patient has every structure delineated; missing structures are simply not present in that
patient's label volume.

NOTE: The raw data also provides a per-voxel dose distribution and a feasible dose mask for the OpenKBP
dose prediction challenge. This module only exposes the OAR/PTV structure masks as segmentation targets;
dose regression is out of scope for this library.

The data is hosted directly (as plain csv files, not through git-lfs) in the challenge repository at
https://github.com/ababier/open-kbp. This module downloads it from a pinned commit of that repository, so
that the checksum of the downloaded archive stays reproducible. It is distributed for research use; see the
repository for the exact license terms.

The CT and structure files each store a flattened (raveled, C order) sparse representation of the
128x128x128 patient volume: only the non-background voxel indices (and, for the CT scan, their values) are
listed. This module reconstructs the dense CT volume and a single-channel structure label volume (using the
priority order of `LABEL_IDS`, so that a PTV overwrites an OAR at an overlapping voxel) and stores them as
hdf5 volumes (keys 'raw' and 'labels') for efficient access.

This dataset is from the publication https://doi.org/10.1002/mp.14845.
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


URL = "https://codeload.github.com/ababier/open-kbp/zip/ce625e62f3b04203f22bd9d1634f3e8fb0245e46"
CHECKSUM = "4074858f3f0d9d7f639342349e2799c1109f3018e3bd487827f334b591ebf697"

COMMIT = "ce625e62f3b04203f22bd9d1634f3e8fb0245e46"

SPLITS = {"train": "train-pats", "val": "validation-pats", "test": "test-pats"}

N_PATIENTS = {"train": 200, "val": 40, "test": 100}

PATIENT_SHAPE = (128, 128, 128)

OARS = ["Brainstem", "SpinalCord", "RightParotid", "LeftParotid", "Esophagus", "Larynx", "Mandible"]
TARGETS = ["PTV56", "PTV63", "PTV70"]
STRUCTURES = OARS + TARGETS

LABEL_IDS = {"background": 0}
LABEL_IDS.update({name: i + 1 for i, name in enumerate(STRUCTURES)})


def _load_sparse_csv(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path, index_col=0)
    indices = df.index.to_numpy()
    if df["data"].isna().all():  # This is a binary structure mask, so it only stores the indices.
        return indices, None
    else:  # This is a sparse volume (eg. the CT scan), so it also stores a value per index.
        return indices, df["data"].to_numpy()


def _preprocess_patient(patient_dir, volume_path):
    import h5py

    raw = np.zeros(np.prod(PATIENT_SHAPE), dtype="int16")
    indices, values = _load_sparse_csv(os.path.join(patient_dir, "ct.csv"))
    raw[indices] = values
    raw = raw.reshape(PATIENT_SHAPE)

    labels = np.zeros(np.prod(PATIENT_SHAPE), dtype="uint8")
    for structure in STRUCTURES:
        structure_path = os.path.join(patient_dir, f"{structure}.csv")
        if not os.path.exists(structure_path):
            continue
        indices, _ = _load_sparse_csv(structure_path)
        labels[indices] = LABEL_IDS[structure]
    labels = labels.reshape(PATIENT_SHAPE)

    with h5py.File(f"{volume_path}.tmp", "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")

    os.rename(f"{volume_path}.tmp", volume_path)


def _preprocess_inputs(data_dir, split, preprocessed_dir):
    patient_dirs = natsorted(glob(os.path.join(data_dir, SPLITS[split], "pt_*")))
    os.makedirs(preprocessed_dir, exist_ok=True)
    for patient_dir in tqdm(patient_dirs, desc=f"Preprocessing the OpenKBP '{split}' patients"):
        volume_path = os.path.join(preprocessed_dir, f"{os.path.basename(patient_dir)}.h5")
        if os.path.exists(volume_path):
            continue
        _preprocess_patient(patient_dir, volume_path)


def get_openkbp_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the OpenKBP dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the provided data is stored.
    """
    data_dir = os.path.join(path, f"open-kbp-{COMMIT}", "provided-data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "open-kbp.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_openkbp_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False
) -> List[str]:
    """Get paths to the OpenKBP data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Please choose one of {list(SPLITS.keys())}.")

    preprocessed_dir = os.path.join(path, "preprocessed", split)
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) != N_PATIENTS[split]:
        data_dir = get_openkbp_data(path, download)
        _preprocess_inputs(data_dir, split, preprocessed_dir)

    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) == N_PATIENTS[split]

    return volume_paths


def get_openkbp_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the OpenKBP dataset for organ-at-risk and planning target volume segmentation.

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
    volume_paths = get_openkbp_paths(path, split, download)

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


def get_openkbp_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the OpenKBP dataloader for organ-at-risk and planning target volume segmentation.

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
    dataset = get_openkbp_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
