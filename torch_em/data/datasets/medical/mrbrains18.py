"""The MRBrainS18 dataset contains annotations for the segmentation of brain structures and of
white matter lesions in multi-sequence brain MRI.

The data was curated for the MRBrainS18 challenge (https://mrbrains18.isi.uu.nl), which was held at MICCAI
2018 and is the successor of the MRBrainS13 challenge. It consists of 30 subjects scanned at the UMC Utrecht
with a 3T scanner, 7 of them released as the training set and the remaining 23 as the test set. The reference
standard was released for the test subjects as well, so this module provides 30 annotated volumes over the
two official splits.

Three co-registered sequences are available per subject and can be selected with the 'modality' argument:
a 3D T1-weighted scan ('t1', registered to the label grid), a multi-slice T1-weighted inversion recovery scan
('ir') and a multi-slice T2 FLAIR scan ('flair'). All of them are bias field corrected with N4ITK and have a
voxel size of 0.958 x 0.958 x 3.0 mm.

The label ids are described in `LABEL_IDS`: 0 = background, 1 = cortical grey matter, 2 = basal ganglia,
3 = white matter, 4 = white matter lesions, 5 = cerebrospinal fluid in the extracerebral space,
6 = ventricles, 7 = cerebellum, 8 = brain stem, 9 = infarction, 10 = other.
NOTE: The official challenge evaluation only scores the ids 1 to 8; the ids 9 and 10 are excluded from it.

The three sequences and the reference standard of a subject are bundled into one hdf5 file per subject by
this module, with the slice axis first (the keys are 'raw/t1', 'raw/ir', 'raw/flair' and 'labels').

The data is hosted at https://doi.org/10.34894/E0U32Q and is free to download, but it may only be used
under the terms of the UMC Utrecht license that is distributed with it.

This dataset is from the publication https://doi.org/10.3389/fncom.2019.00093.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "train": "https://dataverse.nl/api/access/datafile/402709",
    "test": "https://dataverse.nl/api/access/datafile/402708",
}

CHECKSUMS = {
    "train": "c29b133c0d0e7486563d3a92f09e3ae67986880130bb8a91096fc1fdf0c93e95",
    "test": "1576500224bd98ef66fa0277af13bd304ed491974a0de96006f6779e5363d3dd",
}

LABEL_IDS = {
    "background": 0,
    "cortical_grey_matter": 1,
    "basal_ganglia": 2,
    "white_matter": 3,
    "white_matter_lesions": 4,
    "csf_extracerebral": 5,
    "ventricles": 6,
    "cerebellum": 7,
    "brain_stem": 8,
    "infarction": 9,
    "other": 10,
}

SPLITS = {"train": "training", "test": "test"}

MODALITIES = {"t1": "reg_T1.nii.gz", "ir": "reg_IR.nii.gz", "flair": "FLAIR.nii.gz"}

SUBJECT_IDS = {
    "train": [1, 4, 5, 7, 14, 27, 29],
    "test": [2, 3, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 30],
}


def _preprocess_inputs(data_dir, subject_ids, preprocessed_dir):
    import h5py
    import nibabel as nib

    os.makedirs(preprocessed_dir, exist_ok=True)
    for subject_id in tqdm(subject_ids, desc="Preprocessing the MRBrainS18 subjects"):
        volume_path = os.path.join(preprocessed_dir, f"subject_{subject_id:02}.h5")
        if os.path.exists(volume_path):
            continue

        subject_dir = os.path.join(data_dir, str(subject_id))

        # The transpose maps the nifti axis order (X, Y, Z) to the (Z, Y, X) order used for the volumes.
        labels = np.asarray(nib.load(os.path.join(subject_dir, "segm.nii.gz")).dataobj).T

        # The file is written to a temporary path first, so that an interrupted run leaves no corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            for modality, fname in MODALITIES.items():
                raw = np.asarray(nib.load(os.path.join(subject_dir, "pre", fname)).dataobj).T
                f.create_dataset(f"raw/{modality}", data=raw, compression="gzip")

            f.create_dataset("labels", data=labels.astype("uint8"), compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)


def get_mrbrains18_data(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False
) -> str:
    """Download the MRBrainS18 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Please choose one of {list(SPLITS.keys())}.")

    subject_ids = SUBJECT_IDS[split]
    preprocessed_dir = os.path.join(path, "preprocessed", split)
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == len(subject_ids):
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, SPLITS[split])
    if not os.path.exists(data_dir):
        zip_path = os.path.join(path, f"{SPLITS[split]}.zip")
        util.download_source(path=zip_path, url=URLS[split], download=download, checksum=CHECKSUMS[split])
        util.unzip(zip_path=zip_path, dst=path)

    _preprocess_inputs(data_dir, subject_ids, preprocessed_dir)
    return preprocessed_dir


def get_mrbrains18_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "test"]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the MRBrainS18 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' or 'test'. If None, all subjects are returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw/<modality>')
        and the label data ('labels').
    """
    splits = list(SPLITS.keys()) if split is None else [split]
    volume_paths = []
    for curr_split in splits:
        volume_paths.extend(natsorted(glob(os.path.join(get_mrbrains18_data(path, curr_split, download), "*.h5"))))

    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{path}'."
    return volume_paths


def get_mrbrains18_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Optional[Literal["train", "test"]] = None,
    modality: Literal["t1", "ir", "flair"] = "t1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MRBrainS18 dataset for brain structure and white matter lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'. If None, all subjects are returned.
        modality: The MRI sequence. Either 't1', 'ir' or 'flair'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Please choose one of {list(MODALITIES.keys())}.")

    volume_paths = get_mrbrains18_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key=f"raw/{modality}",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_mrbrains18_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Optional[Literal["train", "test"]] = None,
    modality: Literal["t1", "ir", "flair"] = "t1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MRBrainS18 dataloader for brain structure and white matter lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'test'. If None, all subjects are returned.
        modality: The MRI sequence. Either 't1', 'ir' or 'flair'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mrbrains18_dataset(path, patch_shape, split, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
