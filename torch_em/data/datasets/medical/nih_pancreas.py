"""The NIH Pancreas-CT dataset contains annotations for pancreas segmentation in contrast-enhanced abdominal CT.

It consists of 80 CT volumes (the original release had 82 volumes, cases 25 and 70 were removed by TCIA,
as they are duplicates of case 2) with binary pancreas labels. The CT scans are distributed as DICOM series,
which are stacked into volumes and stored together with the corresponding labels in hdf5 files by this module.

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/pancreas-ct/.

This dataset is from the publication https://doi.org/10.1007/978-3-319-24553-9_68.
The data was released at https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU.
Please cite it if you use this dataset in your research.
"""

import os
import csv
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "images": "https://www.cancerimagingarchive.net/wp-content/uploads/Pancreas-CT-20200910.tcia",
    "labels": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_pancreas_labels-02-05-2017-1.zip",
}

CHECKSUMS = {
    "images": None,  # The DICOM series are downloaded individually from TCIA.
    "labels": "cf8a553c37c80e3840ce7392987b308f7a98cd5f50019511b2ac0f3a54b0934b",
}


def _load_dicom_volume(series_dir):
    """Stack a DICOM series into a volume with axes (z, y, x) and slices sorted by ascending patient z position.

    Returns the volume in Hounsfield units and the image orientation (DICOM 'ImageOrientationPatient').
    """
    import pydicom

    slices = [pydicom.dcmread(dcm_path) for dcm_path in natsorted(glob(os.path.join(series_dir, "*.dcm")))]
    slices.sort(key=lambda dcm: float(dcm.ImagePositionPatient[2]))

    volume = np.stack([dcm.pixel_array for dcm in slices]).astype("float32")
    volume = volume * float(slices[0].RescaleSlope) + float(slices[0].RescaleIntercept)
    volume = np.round(volume).astype("int16")

    orientation = np.round([float(v) for v in slices[0].ImageOrientationPatient]).astype("int").tolist()
    return volume, orientation


def _preprocess_nih_pancreas(dicom_dir, label_dir, csv_path, preprocessed_dir):
    import h5py
    import nibabel as nib

    with open(csv_path, "r") as f:
        subject_ids = {row["Series UID"]: row["Subject ID"] for row in csv.DictReader(f)}

    os.makedirs(preprocessed_dir, exist_ok=True)
    for series_dir in tqdm(natsorted(glob(os.path.join(dicom_dir, "*"))), desc="Preprocess NIH Pancreas-CT"):
        subject_id = subject_ids[os.path.basename(series_dir)]
        out_path = os.path.join(preprocessed_dir, f"{subject_id}.h5")
        if os.path.exists(out_path):
            continue

        volume, orientation = _load_dicom_volume(series_dir)
        # The volume has axes (z, y, x) with x pointing to the patient's left and y to the anterior (DICOM LPS
        # convention with 'ImageOrientationPatient' [1, 0, 0, 0, -1, 0]). The labels are stored with the axis
        # orientation (L, A, I), so they only have to be transposed and flipped along z to match the volume.
        assert orientation == [1, 0, 0, 0, -1, 0], f"Unexpected image orientation for {subject_id}: {orientation}"

        label_path = os.path.join(label_dir, f"label{subject_id.split('_')[-1]}.nii.gz")
        label_nifti = nib.load(label_path)
        assert nib.aff2axcodes(label_nifti.affine) == ("L", "A", "I"), f"Unexpected label axes for {subject_id}"
        labels = np.asarray(label_nifti.dataobj).astype("uint8").transpose(2, 1, 0)[::-1]
        assert labels.shape == volume.shape, f"Shape mismatch for {subject_id}: {labels.shape} vs {volume.shape}"

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_nih_pancreas_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the NIH Pancreas-CT dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if os.path.exists(preprocessed_dir):
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    # Download the labels.
    label_dir = os.path.join(path, "TCIA_pancreas_labels-02-05-2017")
    if not os.path.exists(label_dir):
        zip_path = os.path.join(path, "TCIA_pancreas_labels-02-05-2017-1.zip")
        util.download_source(path=zip_path, url=URLS["labels"], download=download, checksum=CHECKSUMS["labels"])
        util.unzip(zip_path=zip_path, dst=path)

    # Download the DICOM series from the TCIA manifest.
    dicom_dir = os.path.join(path, "dicom")
    csv_path = os.path.join(path, "nih_pancreas_series")
    util.download_source_tcia(
        path=os.path.join(path, "Pancreas-CT-20200910.tcia"), url=URLS["images"], dst=dicom_dir,
        csv_filename=csv_path, download=download,
    )

    _preprocess_nih_pancreas(dicom_dir, label_dir, f"{csv_path}.csv", preprocessed_dir)
    return preprocessed_dir


def get_nih_pancreas_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the NIH Pancreas-CT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_nih_pancreas_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_nih_pancreas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the NIH Pancreas-CT dataset for pancreas segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_nih_pancreas_paths(path, download)

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


def get_nih_pancreas_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the NIH Pancreas-CT dataloader for pancreas segmentation.

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
    dataset = get_nih_pancreas_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
