"""The CT Lymph Nodes dataset contains annotations for lymph node segmentation in mediastinal and abdominal CT.

It consists of 176 CT volumes (90 mediastinal and 86 abdominal scans) with manually traced lymph node
segmentations. The labels are instance labels, i.e. each lymph node has its own id. The CT scans are distributed
as DICOM series (ca. 58 GB) and the segmentation masks as nifti files, which are stacked, aligned and stored
together in hdf5 files by this module.

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/ct-lymph-nodes/.

This dataset is from the publications https://doi.org/10.1007/978-3-319-10404-1_65 and
https://doi.org/10.1007/978-3-319-24571-3_7 (segmentation masks).
The data was released at https://doi.org/10.7937/K9/TCIA.2015.AQIIDCNM.
Please cite it if you use this dataset in your research.
"""

import os
import csv
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "images": "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_CT_Lymph_Nodes_06-22-2015.tcia",
    "labels": "https://www.cancerimagingarchive.net/wp-content/uploads/MED_ABD_LYMPH_MASKS.zip",
}

CHECKSUMS = {
    "images": None,  # The DICOM series are downloaded individually from TCIA.
    "labels": "ace3475c21f04c3f3a01e7fa5181fcbcf4a98cc78ea945e058f1b001d25d6745",
}

REGIONS = {"mediastinal": "MED", "abdominal": "ABD"}


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


def _preprocess_ct_lymph_nodes(dicom_dir, label_dir, csv_path, preprocessed_dir):
    import h5py
    import nibabel as nib

    with open(csv_path, "r") as f:
        subject_ids = {row["Subject ID"]: row["Series UID"] for row in csv.DictReader(f) if row["Modality"] == "CT"}

    os.makedirs(preprocessed_dir, exist_ok=True)
    for subject_id, series_uid in tqdm(sorted(subject_ids.items()), desc="Preprocess CT Lymph Nodes"):
        out_path = os.path.join(preprocessed_dir, f"{subject_id}.h5")
        if os.path.exists(out_path):
            continue

        volume, orientation = _load_dicom_volume(os.path.join(dicom_dir, series_uid))
        # The volume has axes (z, y, x) with x pointing to the patient's left and y to the posterior (DICOM LPS
        # convention with 'ImageOrientationPatient' [1, 0, 0, 0, 1, 0]). The labels are stored with the axis
        # orientation (L, P, S), so they only have to be transposed to match the volume.
        assert orientation == [1, 0, 0, 0, 1, 0], f"Unexpected image orientation for {subject_id}: {orientation}"

        label_nifti = nib.load(os.path.join(label_dir, subject_id, f"{subject_id}_mask.nii.gz"))
        assert nib.aff2axcodes(label_nifti.affine) == ("L", "P", "S"), f"Unexpected label axes for {subject_id}"
        labels = np.asarray(label_nifti.dataobj).astype("uint8").transpose(2, 1, 0)
        assert labels.shape == volume.shape, f"Shape mismatch for {subject_id}: {labels.shape} vs {volume.shape}"

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_ct_lymph_nodes_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CT Lymph Nodes dataset.

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
    label_dir = os.path.join(path, "MED_ABD_LYMPH_MASKS")
    if not os.path.exists(label_dir):
        zip_path = os.path.join(path, "MED_ABD_LYMPH_MASKS.zip")
        util.download_source(path=zip_path, url=URLS["labels"], download=download, checksum=CHECKSUMS["labels"])
        util.unzip(zip_path=zip_path, dst=path)

    # Download the DICOM series from the TCIA manifest.
    dicom_dir = os.path.join(path, "dicom")
    csv_path = os.path.join(path, "ct_lymph_nodes_series")
    util.download_source_tcia(
        path=os.path.join(path, "TCIA_CT_Lymph_Nodes_06-22-2015.tcia"), url=URLS["images"], dst=dicom_dir,
        csv_filename=csv_path, download=download,
    )

    _preprocess_ct_lymph_nodes(dicom_dir, label_dir, f"{csv_path}.csv", preprocessed_dir)
    return preprocessed_dir


def get_ct_lymph_nodes_paths(
    path: Union[os.PathLike, str],
    region: Optional[Literal["mediastinal", "abdominal"]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the CT Lymph Nodes data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        region: The choice of body region. Either 'mediastinal' or 'abdominal'. If None, all volumes are returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_ct_lymph_nodes_data(path, download)

    if region is None:
        prefix = "*"
    elif region in REGIONS:
        prefix = REGIONS[region]
    else:
        raise ValueError(f"'{region}' is not a valid region. Please choose one of {list(REGIONS.keys())}.")

    volume_paths = natsorted(glob(os.path.join(data_dir, f"{prefix}_LYMPH_*.h5")))
    return volume_paths


def get_ct_lymph_nodes_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    region: Optional[Literal["mediastinal", "abdominal"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CT Lymph Nodes dataset for lymph node segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        region: The choice of body region. Either 'mediastinal' or 'abdominal'. If None, all volumes are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_ct_lymph_nodes_paths(path, region, download)

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


def get_ct_lymph_nodes_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    region: Optional[Literal["mediastinal", "abdominal"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CT Lymph Nodes dataloader for lymph node segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        region: The choice of body region. Either 'mediastinal' or 'abdominal'. If None, all volumes are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ct_lymph_nodes_dataset(path, patch_shape, region, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
