"""The MS3SEG dataset contains annotations for three-class segmentation of the ventricles, normal
age-related white matter hyperintensities (WMH) and pathological multiple sclerosis (MS) WMH lesions
in axial T2-FLAIR brain MRI.

The dataset consists of 100 MS patients acquired on a 1.5T Toshiba scanner, with T1-weighted,
T2-weighted, and axial / sagittal T2-FLAIR sequences. Expert annotators delineated three classes on the
axial T2-FLAIR images: ventricles, normal WMH and abnormal (pathological) WMH.

NOTE: The data is distributed as password-free RAR archives, which requires the 'p7zip' CLI (or the
'rarfile' python package) to extract.

This dataset is from the publication https://doi.org/10.1038/s41597-026-07184-5.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "nifti_part1": "https://ndownloader.figshare.com/files/61900798",
    "nifti_part2": "https://ndownloader.figshare.com/files/61901377",
    "nifti_part3": "https://ndownloader.figshare.com/files/61901674",
    "masks": "https://ndownloader.figshare.com/files/65733546",
}

CHECKSUMS = {
    "nifti_part1": "6a7499b8a6496b76de13783c43af6194b7e81ac1130137a8db639f51a29a973e",
    "nifti_part2": "2637a7971e80527f752edfed9e8674daec454b2f020573e1a6279b895f46c7fe",
    "nifti_part3": "9b3c71edb700b424a502741623bed38bd3ebaba7dacaef1c4935efc636661e62",
    "masks": "6c5d2fddc5ed89988e8c15f060e1564e9ad5e7a10ed4da45070ace397e5c594c",
}

LABEL_IDS = {"background": 0, "ventricle": 1, "normal_wmh": 2, "ms_wmh": 3}


def _preprocess_inputs(path, nifti_dir, masks_dir, preprocessed_dir):
    import h5py
    import nibabel as nib

    os.makedirs(preprocessed_dir, exist_ok=True)

    case_dirs = [p for p in natsorted(glob(os.path.join(nifti_dir, "*"))) if os.path.isdir(p)]
    for case_dir in tqdm(case_dirs, desc="Preprocessing the MS3SEG cases"):
        case_id = os.path.basename(case_dir)
        volume_path = os.path.join(preprocessed_dir, f"{case_id}.h5")
        if os.path.exists(volume_path):
            continue

        raw_path = os.path.join(case_dir, f"{case_id}_FLAIR.nii.gz")
        vent_path = os.path.join(masks_dir, "Vent_Masks", case_id, f"{case_id}_Vent_Mask.nii.gz")
        nwmh_path = os.path.join(masks_dir, "nWMH_Masks", case_id, f"{case_id}_nWMH_Mask.nii.gz")
        abwmh_path = os.path.join(masks_dir, "abWMH_Masks", case_id, f"{case_id}_abWMH_Mask.nii.gz")
        if not all(os.path.exists(p) for p in (raw_path, vent_path, nwmh_path, abwmh_path)):
            continue

        raw = np.asarray(nib.load(raw_path).dataobj)
        vent = np.asarray(nib.load(vent_path).dataobj)
        nwmh = np.asarray(nib.load(nwmh_path).dataobj)
        abwmh = np.asarray(nib.load(abwmh_path).dataobj)

        # The abnormal (MS) WMH class takes priority over the normal WMH class, which in turn
        # takes priority over the ventricle class, in case of overlapping annotations.
        labels = np.zeros(raw.shape, dtype="uint8")
        labels[vent > 0] = LABEL_IDS["ventricle"]
        labels[nwmh > 0] = LABEL_IDS["normal_wmh"]
        labels[abwmh > 0] = LABEL_IDS["ms_wmh"]

        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)


def get_ms3seg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the MS3SEG dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if os.path.exists(preprocessed_dir) and len(glob(os.path.join(preprocessed_dir, "*.h5"))) > 0:
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    nifti_dir = os.path.join(path, "MS_100_patient_nifti")
    masks_dir = os.path.join(path, "MS_100_patient_masks")

    if not os.path.exists(nifti_dir):
        for name in ("nifti_part1", "nifti_part2", "nifti_part3"):
            rar_path = os.path.join(path, f"{name}.rar")
            util.download_source(path=rar_path, url=URLS[name], download=download, checksum=CHECKSUMS[name])
        util.unzip_rarfile(rar_path=os.path.join(path, "nifti_part1.rar"), dst=path, remove=False)

    if not os.path.exists(masks_dir):
        rar_path = os.path.join(path, "masks.rar")
        util.download_source(path=rar_path, url=URLS["masks"], download=download, checksum=CHECKSUMS["masks"])
        util.unzip_rarfile(rar_path=rar_path, dst=path, remove=False)

    _preprocess_inputs(path, nifti_dir, masks_dir, preprocessed_dir)
    return preprocessed_dir


def get_ms3seg_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the MS3SEG data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_ms3seg_data(path, download)
    return natsorted(glob(os.path.join(data_dir, "*.h5")))


def get_ms3seg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MS3SEG dataset for three-class segmentation of ventricles, normal WMH and MS lesions.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_ms3seg_paths(path, download)

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


def get_ms3seg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MS3SEG dataloader for three-class segmentation of ventricles, normal WMH and MS lesions.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ms3seg_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
