"""The ISBI MS Lesion dataset contains annotations for multiple sclerosis lesion segmentation
in longitudinal brain MRI.

The data was curated for the Longitudinal Multiple Sclerosis Lesion Segmentation Challenge
(https://iacl.ece.jhu.edu/index.php/MSChallenge), which was held at ISBI 2015. The public training release
consists of 5 subjects that were scanned at 4 to 5 time points each, which amounts to 21 time points, and
each time point was delineated independently by 2 raters. This module therefore provides 21 volumes with
2 sets of annotations, i.e. 42 annotated volumes, which are exposed together with the 'both' choice of the
'rater' argument. The 14 test subjects (61 further time points) are distributed without annotations and are
therefore not exposed here.

Four co-registered sequences are available per time point and can be selected with the 'modality' argument:
a T2 FLAIR scan ('flair'), a T1-weighted MPRAGE scan ('mprage'), a proton density scan ('pd') and a
T2-weighted scan ('t2'). This module uses the preprocessed version of the scans (inhomogeneity corrected,
skull stripped and rigidly registered to a 1 mm isotropic MNI template), since only it is aligned with the
lesion masks. The scans in native acquisition space are also part of the download, but are not used here.

The annotations are binary, see `LABEL_IDS`: 1 = multiple sclerosis lesion.

The four sequences and the two sets of annotations of a time point are bundled into one hdf5 file per time
point by this module, with the slice axis first (the keys are 'raw/flair', 'raw/mprage', 'raw/pd', 'raw/t2',
'labels/rater1' and 'labels/rater2').

The data is located at https://iacl.ece.jhu.edu/index.php/MSChallenge/data and may only be used for research
and education, see the license that is distributed with it.

This dataset is from the publication https://doi.org/10.1016/j.neuroimage.2016.12.064.
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

from ... import ConcatDataset
from .. import util


URL = "https://iacl.ece.jhu.edu/~aaron/data/training_final_v4.zip"
CHECKSUM = "f5db0c71d0dd90a19b156f8ab7eb6081dc51571c78c06646a7200bbb2b090be6"

LABEL_IDS = {"background": 0, "ms_lesion": 1}

MODALITIES = ["flair", "mprage", "pd", "t2"]

RATERS = [1, 2]

# The number of time points per training subject. They add up to the 21 time points of the training release.
TIME_POINTS = {"training01": 4, "training02": 4, "training03": 5, "training04": 4, "training05": 4}


def _preprocess_inputs(data_dir, preprocessed_dir):
    import h5py
    import nibabel as nib

    os.makedirs(preprocessed_dir, exist_ok=True)
    for subject_id, n_time_points in tqdm(TIME_POINTS.items(), desc="Preprocessing the ISBI MS Lesion subjects"):
        for time_point in range(1, n_time_points + 1):
            case_id = f"{subject_id}_{time_point:02}"
            volume_path = os.path.join(preprocessed_dir, f"{case_id}.h5")
            if os.path.exists(volume_path):
                continue

            # The transpose maps the nifti axis order (X, Y, Z) to the (Z, Y, X) order used for the volumes.
            # The file is written to a temporary path first, so an interrupted run leaves no corrupt file.
            with h5py.File(f"{volume_path}.tmp", "w") as f:
                for modality in MODALITIES:
                    raw_path = os.path.join(data_dir, subject_id, "preprocessed", f"{case_id}_{modality}_pp.nii")
                    raw = np.asarray(nib.load(raw_path).dataobj).T
                    f.create_dataset(f"raw/{modality}", data=raw, compression="gzip")

                for rater in RATERS:
                    label_path = os.path.join(data_dir, subject_id, "masks", f"{case_id}_mask{rater}.nii")
                    labels = np.asarray(nib.load(label_path).dataobj).T > 0
                    f.create_dataset(f"labels/rater{rater}", data=labels.astype("uint8"), compression="gzip")

            os.rename(f"{volume_path}.tmp", volume_path)


def get_isbi_mslesion_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ISBI MS Lesion dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == sum(TIME_POINTS.values()):
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "training")
    if not os.path.exists(data_dir):
        zip_path = os.path.join(path, "training_final_v4.zip")
        util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip(zip_path=zip_path, dst=path, remove=False)

    _preprocess_inputs(data_dir, preprocessed_dir)
    return preprocessed_dir


def get_isbi_mslesion_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the ISBI MS Lesion data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw/<modality>')
        and the label data ('labels/rater<rater>').
    """
    data_dir = get_isbi_mslesion_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "training*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{data_dir}'."
    return volume_paths


def get_isbi_mslesion_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Literal["flair", "mprage", "pd", "t2"] = "flair",
    rater: Literal[1, 2, "both"] = "both",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ISBI MS Lesion dataset for multiple sclerosis lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The MRI sequence. Either 'flair', 'mprage', 'pd' or 't2'.
        rater: The choice of annotator. Either 1, 2 or 'both', which returns the volumes of both raters.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Please choose one of {MODALITIES}.")

    if rater not in RATERS and rater != "both":
        raise ValueError(f"'{rater}' is not a valid rater. Please choose one of {RATERS} or 'both'.")

    volume_paths = get_isbi_mslesion_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    datasets = []
    for curr_rater in (RATERS if rater == "both" else [rater]):
        datasets.append(
            torch_em.default_segmentation_dataset(
                raw_paths=volume_paths,
                raw_key=f"raw/{modality}",
                label_paths=volume_paths,
                label_key=f"labels/rater{curr_rater}",
                patch_shape=patch_shape,
                is_seg_dataset=True,
                **kwargs
            )
        )

    return datasets[0] if len(datasets) == 1 else ConcatDataset(*datasets)


def get_isbi_mslesion_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Literal["flair", "mprage", "pd", "t2"] = "flair",
    rater: Literal[1, 2, "both"] = "both",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ISBI MS Lesion dataloader for multiple sclerosis lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The MRI sequence. Either 'flair', 'mprage', 'pd' or 't2'.
        rater: The choice of annotator. Either 1, 2 or 'both', which returns the volumes of both raters.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_isbi_mslesion_dataset(path, patch_shape, modality, rater, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
