"""The Mindboggle-101 dataset contains annotations for cortical parcellation in T1-weighted brain MRI.

The dataset comprises 101 healthy subjects pooled from five cohorts (OASIS-TRT-20, NKI-TRT-20, NKI-RS-22,
MMRR-21 and Extra-18, the last of which bundles the smaller HLN-12, Twins-2, MMRR-3T7T-2, Colin27 and
Afterthought cohorts) that were manually labelled according to the Desikan-Killiany-Tourville (DKT)
cortical labeling protocol.

The data is hosted at https://zenodo.org/records/22070005 (DOI 10.5281/zenodo.22070005) and distributed
under the CC BY 4.0 license, no signup required. This module downloads and uses only the
'Mindboggle101_release3.zip' archive, which stores the labelled volumes as one tar.gz file per cohort
(the archive also ships unlabelled templates, atlases and cortical surfaces, which are not used here).

Two label sets are available per subject and can be selected with the 'label_choice' argument:
- 'cortical': `labels.DKT31.manual.nii.gz`, the manually corrected DKT31 cortical parcellation
  (background plus 31 cortical regions per hemisphere).
- 'full' (default): `labels.DKT31.manual+aseg.nii.gz`, the same cortical labels combined with FreeSurfer's
  automated subcortical segmentation ('aseg'), giving a whole-brain parcellation into over 100 structures.
NOTE: Only the 'cortical' labels are the result of manual editing; the subcortical structures added in the
'full' label set come from FreeSurfer's automated 'aseg' pipeline and are not manually verified.

The T1 volume and the chosen label volume of each subject are bundled into one hdf5 file by this module,
with the keys 'raw' and 'labels'.

This dataset is from the publication https://doi.org/10.3389/fnins.2012.00171.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/22070005/files/Mindboggle101_release3.zip?download=1"

CHECKSUM = "56dfca8fed2f80740c763c0bf05e48cc736eb793ee1511f621ef5f82eb08f26e"

COHORTS = ["OASIS-TRT-20", "NKI-TRT-20", "NKI-RS-22", "MMRR-21", "Extra-18"]

LABEL_FILES = {"cortical": "labels.DKT31.manual.nii.gz", "full": "labels.DKT31.manual+aseg.nii.gz"}


def _get_volumes_dir(path: Union[os.PathLike, str], download: bool) -> str:
    import zipfile

    volumes_dir = os.path.join(path, "Mindboggle101_release3", "Mindboggle101_volumes")
    if all(os.path.exists(os.path.join(volumes_dir, f"{cohort}_volumes")) for cohort in COHORTS):
        return volumes_dir

    os.makedirs(path, exist_ok=True)

    tar_paths = {cohort: os.path.join(volumes_dir, f"{cohort}_volumes.tar.gz") for cohort in COHORTS}
    if not all(os.path.exists(p) for p in tar_paths.values()):
        zip_path = os.path.join(path, "Mindboggle101_release3.zip")
        util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)

        tar_names = [f"Mindboggle101_release3/Mindboggle101_volumes/{cohort}_volumes.tar.gz" for cohort in COHORTS]
        with zipfile.ZipFile(zip_path) as f:
            f.extractall(path, members=tar_names)

        os.remove(zip_path)

    for cohort in COHORTS:
        target_dir = os.path.join(volumes_dir, f"{cohort}_volumes")
        if os.path.exists(target_dir):
            continue
        util.unzip_tarfile(tar_paths[cohort], dst=volumes_dir, remove=True)

    return volumes_dir


def _preprocess_inputs(volumes_dir, label_choice, preprocessed_dir):
    import h5py
    import nibabel as nib

    os.makedirs(preprocessed_dir, exist_ok=True)
    label_file = LABEL_FILES[label_choice]

    subject_dirs = natsorted(glob(os.path.join(volumes_dir, "*_volumes", "*")))
    for subject_dir in tqdm(subject_dirs, desc="Preprocessing the Mindboggle-101 subjects"):
        subject_id = os.path.basename(subject_dir)
        volume_path = os.path.join(preprocessed_dir, f"{subject_id}.h5")
        if os.path.exists(volume_path):
            continue

        raw_path = os.path.join(subject_dir, "t1weighted.nii.gz")
        label_path = os.path.join(subject_dir, label_file)
        if not (os.path.exists(raw_path) and os.path.exists(label_path)):
            continue

        raw = np.asarray(nib.load(raw_path).dataobj)
        labels = np.asarray(nib.load(label_path).dataobj).astype("uint16")
        assert raw.shape == labels.shape, f"Shape mismatch for {subject_id}: {raw.shape} vs. {labels.shape}."

        # The file is written to a temporary path first, so that an interrupted run leaves no corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)

    # Marks that preprocessing has finished for all subjects, so that a partially preprocessed
    # directory (e.g. from an interrupted previous run) is not mistaken for a complete one.
    open(os.path.join(preprocessed_dir, ".preprocessing_done"), "w").close()


def get_mindboggle101_data(
    path: Union[os.PathLike, str], label_choice: Literal["cortical", "full"] = "full", download: bool = False
) -> str:
    """Download the Mindboggle-101 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        label_choice: The choice of label set. Either 'cortical' or 'full'. See the module docstring.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the preprocessed data is stored.
    """
    if label_choice not in LABEL_FILES:
        raise ValueError(f"'{label_choice}' is not a valid label choice. Please choose one of {list(LABEL_FILES)}.")

    preprocessed_dir = os.path.join(path, "preprocessed", label_choice)
    if os.path.exists(os.path.join(preprocessed_dir, ".preprocessing_done")):
        return preprocessed_dir

    volumes_dir = _get_volumes_dir(path, download)
    _preprocess_inputs(volumes_dir, label_choice, preprocessed_dir)
    return preprocessed_dir


def get_mindboggle101_paths(
    path: Union[os.PathLike, str],
    label_choice: Literal["cortical", "full"] = "full",
    download: bool = False,
) -> List[str]:
    """Get paths to the Mindboggle-101 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        label_choice: The choice of label set. Either 'cortical' or 'full'. See the module docstring.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    preprocessed_dir = get_mindboggle101_data(path, label_choice, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed Mindboggle-101 volumes at '{path}'."
    return volume_paths


def get_mindboggle101_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    label_choice: Literal["cortical", "full"] = "full",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Mindboggle-101 dataset for cortical parcellation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of label set. Either 'cortical' or 'full'. See the module docstring.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_mindboggle101_paths(path, label_choice, download)

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


def get_mindboggle101_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    label_choice: Literal["cortical", "full"] = "full",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Mindboggle-101 dataloader for cortical parcellation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of label set. Either 'cortical' or 'full'. See the module docstring.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mindboggle101_dataset(path, patch_shape, label_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
