"""The VALDO dataset contains annotations for cerebral microbleed segmentation in brain MRI.

The data was curated for Task 2 of the VALDO challenge ('Where is VALDO? VAscular Lesions DetectiOn and
segmentation', https://valdo.grand-challenge.org/Task2), which was held at MICCAI 2021 and targets the
detection and segmentation of small vessel disease markers. The public training release of Task 2 consists
of 72 subjects pooled from three cohorts (the subject ids 'sub-1xx', 'sub-2xx' and 'sub-3xx' correspond to
11, 34 and 27 subjects respectively), so this module provides 72 annotated volumes. The test data of the
challenge was never released.

Three co-registered sequences are available per subject and can be selected with the 'modality' argument:
the T2* scan ('t2s'), a T2-weighted scan ('t2') and a T1-weighted scan ('t1'). All of them are resampled to
the T2* space, in which the microbleeds are annotated, and the voxel size differs between the three cohorts.
NOTE: The T1 and T2 volumes have NaNs in the masked-out background, which are set to zero by this module.

The annotations are binary, see `LABEL_IDS`: 1 = cerebral microbleed. The challenge evaluation derives
individual lesions from them with a connected component analysis with a neighbourhood of 6, so instance
labels can be obtained with `scipy.ndimage.label`.

The three sequences and the annotations of a subject are bundled into one hdf5 file per subject by this
module, with the slice axis first (the keys are 'raw/t2s', 'raw/t2', 'raw/t1' and 'labels').

The data is located at https://doi.org/10.5281/zenodo.4687995 and is licensed under CC BY-NC-SA 4.0.

This dataset is from the publication https://doi.org/10.48550/arXiv.2208.07167.
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


URL = "https://zenodo.org/records/4687995/files/Task2_v2.tar.gz"
CHECKSUM = "d47a9104deffd6a3a813ae53bf61209f734cce2818aa11611c1daf71188806b4"

LABEL_IDS = {"background": 0, "microbleed": 1}

MODALITIES = {"t2s": "T2S", "t2": "T2", "t1": "T1"}


def _preprocess_inputs(data_dir, preprocessed_dir):
    import h5py
    import nibabel as nib

    subject_dirs = natsorted(glob(os.path.join(data_dir, "sub-*")))
    os.makedirs(preprocessed_dir, exist_ok=True)

    for subject_dir in tqdm(subject_dirs, desc="Preprocessing the VALDO subjects"):
        subject_id = os.path.basename(subject_dir)
        volume_path = os.path.join(preprocessed_dir, f"{subject_id}.h5")
        if os.path.exists(volume_path):
            continue

        # The transpose maps the nifti axis order (X, Y, Z) to the (Z, Y, X) order used for the volumes.
        label_path = os.path.join(subject_dir, f"{subject_id}_space-T2S_CMB.nii.gz")
        labels = np.asarray(nib.load(label_path).dataobj).T > 0

        # The file is written to a temporary path first, so that an interrupted run leaves no corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            for modality, suffix in MODALITIES.items():
                raw_path = os.path.join(subject_dir, f"{subject_id}_space-T2S_desc-masked_{suffix}.nii.gz")
                # The background of the T1 and T2 scans is NaN, which would propagate into the normalization.
                raw = np.nan_to_num(np.asarray(nib.load(raw_path).dataobj, dtype="float32").T)
                f.create_dataset(f"raw/{modality}", data=raw, compression="gzip")

            f.create_dataset("labels", data=labels.astype("uint8"), compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)


def get_valdo_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the VALDO Task 2 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == 72:
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "Task2")
    if not os.path.exists(data_dir):
        tar_path = os.path.join(path, "Task2_v2.tar.gz")
        util.download_source(path=tar_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip_tarfile(tar_path=tar_path, dst=path, remove=False)

    _preprocess_inputs(data_dir, preprocessed_dir)
    return preprocessed_dir


def get_valdo_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the VALDO Task 2 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw/<modality>')
        and the label data ('labels').
    """
    data_dir = get_valdo_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "sub-*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{data_dir}'."
    return volume_paths


def get_valdo_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Literal["t2s", "t2", "t1"] = "t2s",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the VALDO Task 2 dataset for cerebral microbleed segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The MRI sequence. Either 't2s', 't2' or 't1'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Please choose one of {list(MODALITIES.keys())}.")

    volume_paths = get_valdo_paths(path, download)

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


def get_valdo_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Literal["t2s", "t2", "t1"] = "t2s",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the VALDO Task 2 dataloader for cerebral microbleed segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The MRI sequence. Either 't2s', 't2' or 't1'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_valdo_dataset(path, patch_shape, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
