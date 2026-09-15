"""The ACRIN-HNSCC-FDG-PET-CT dataset contains annotations for head and neck tumor and lymph node segmentation
in FDG-PET and CT of head and neck squamous cell carcinoma patients from the ACRIN 6685 trial.

The lesion segmentations were created by radiologists in the 'ACRIN 6685-Tumor-Annotations' analysis result
of the collection and are distributed as DICOM RTSTRUCT (one file per lesion, plus a seed point file per lesion
and 'negative' assessments for scans without findings). They reference 257 PET, 378 CT and 31 MR series.
The RTSTRUCT annotations are public, but the images are only available under the NIH Controlled Data Access
Policy and have to be downloaded manually, see `get_acrin_hnscc_data`. This module rasterizes the lesion contours
onto the image grid (see `torch_em.data.datasets.util.rasterize_rtstruct`) and stores images and labels in hdf5
files, one per annotated PET or CT series. The labels are instance labels: each annotated lesion (primary tumor
or lymph node) gets its own id, starting from 1.

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/acrin-hnscc-fdg-pet-ct/ and the annotations
at https://www.cancerimagingarchive.net/analysis-result/acrin-6685-tumor-annotations/.

This dataset is from the publication https://doi.org/10.1200/JCO.18.01182.
The data was released at https://doi.org/10.7937/K9/TCIA.2016.JQEJZZNG (images) and https://doi.org/10.7937/jvgc-aq36
(annotations). Please cite them if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from collections import defaultdict
from typing import Union, Tuple, List, Literal

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "annotations": "https://www.cancerimagingarchive.net/wp-content/uploads/ACRIN6685_Tumor-Annotations-manifest_2026-01-08.tcia",  # noqa
    "metadata": "https://www.cancerimagingarchive.net/wp-content/uploads/Metadata_Report_ACRIN6685_2026-01-08.csv",
}

CHECKSUMS = {
    "annotations": None,  # The DICOM series are downloaded individually from TCIA.
    "metadata": "2462ec93293e8a4d9fdf1c1ae7c6badb0766a383507ea2947c4abb5b6223a4e0",
}

MODALITIES = {"PET": "PT", "CT": "CT"}


def _find_image_series(image_dir):
    """Map the series instance UIDs to the folders with the DICOM files, independent of the folder layout."""
    import pydicom

    series_dirs = {}
    for root, _, files in os.walk(image_dir):
        dcm_files = [fname for fname in files if fname.endswith(".dcm")]
        if dcm_files:
            header = pydicom.dcmread(os.path.join(root, dcm_files[0]), stop_before_pixels=True)
            series_dirs[str(header.SeriesInstanceUID)] = root
    return series_dirs


def _preprocess_acrin_hnscc(annotation_dir, image_dir, metadata_path, preprocessed_dir, modality):
    import h5py

    metadata = pd.read_csv(metadata_path)
    metadata = metadata[
        (metadata["AnnotationType"] == "Segmentation") &
        (metadata["ReferencedSeriesModality"] == MODALITIES[modality])
    ]
    lesions_per_series = defaultdict(list)
    for _, row in metadata.iterrows():
        lesions_per_series[row["ReferencedSeriesInstanceUID"]].append(row)

    image_series = _find_image_series(image_dir)
    missing = [uid for uid in lesions_per_series if uid not in image_series]
    if missing:
        print(f"{len(missing)} of {len(lesions_per_series)} annotated {modality} series were not found in {image_dir}.")

    os.makedirs(preprocessed_dir, exist_ok=True)
    for series_uid, lesions in tqdm(sorted(lesions_per_series.items()), desc=f"Preprocess ACRIN-HNSCC {modality}"):
        if series_uid not in image_series:
            continue
        patient_id = lesions[0]["PatientID"]
        time_point = lesions[0]["ClinicalTrialTimePointID"].lower().replace(" ", "_").replace("#", "")
        out_path = os.path.join(preprocessed_dir, f"{patient_id}_{time_point}_{series_uid[-8:]}.h5")
        if os.path.exists(out_path):
            continue

        volume, geometry = util.load_dicom_series(image_series[series_uid])
        if modality == "CT":
            volume = np.round(volume).astype("int16")

        labels = np.zeros(volume.shape, dtype="uint8")
        for lesion_id, lesion in enumerate(sorted(lesions, key=lambda row: row["TrackingID"]), start=1):
            rtstruct_path = glob(os.path.join(annotation_dir, lesion["SeriesInstanceUID"], "*.dcm"))[0]
            lesion_mask = util.rasterize_rtstruct(
                rtstruct_path, geometry, volume.shape, lambda roi_number, roi_name: 1
            )
            labels[lesion_mask > 0] = lesion_id

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_acrin_hnscc_data(
    path: Union[os.PathLike, str], modality: Literal["PET", "CT"], download: bool = False
) -> str:
    """Download the ACRIN-HNSCC-FDG-PET-CT dataset.

    The RTSTRUCT annotations and their metadata are downloaded automatically. The images are only available
    under the NIH Controlled Data Access Policy and cannot be downloaded automatically. To obtain them:
    1. Request access following https://www.cancerimagingarchive.net/access-data/ (NIH Controlled Data Access).
    2. On https://www.cancerimagingarchive.net/analysis-result/acrin-6685-tumor-annotations/ download the manifest
       'Original ACRIN 6685 Images used to create Segmentations and Seed Points' (CT, PT, MR, 39.9 GB) and open it
       with the NBIA Data Retriever, logged in with your account.
    3. Store the downloaded DICOM series in the folder '<path>/images'. Any folder layout works, as the series are
       identified by the 'SeriesInstanceUID' in the DICOM headers.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The imaging modality. One of 'PET' or 'CT'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    assert modality in MODALITIES, f"'{modality}' is not a valid modality. Choose one of {list(MODALITIES)}."
    # NOTE: The preprocessing below skips volumes that were converted already, so an interrupted run resumes.
    preprocessed_dir = os.path.join(path, "preprocessed", modality)
    os.makedirs(path, exist_ok=True)

    # Download the public RTSTRUCT annotations and the annotation metadata.
    annotation_dir = os.path.join(path, "annotations")
    csv_path = os.path.join(path, "acrin_hnscc_annotations")
    if not os.path.exists(f"{csv_path}.csv"):
        util.download_source_tcia(
            path=os.path.join(path, os.path.basename(URLS["annotations"])), url=URLS["annotations"],
            dst=annotation_dir, csv_filename=csv_path, download=download,
        )
    metadata_path = os.path.join(path, os.path.basename(URLS["metadata"]))
    util.download_source(path=metadata_path, url=URLS["metadata"], download=download, checksum=CHECKSUMS["metadata"])

    image_dir = os.path.join(path, "images")
    if not os.path.exists(image_dir):
        raise RuntimeError(
            f"The ACRIN-HNSCC-FDG-PET-CT images were not found at {image_dir}. They are only available under the "
            "NIH Controlled Data Access Policy and have to be downloaded manually: request access at "
            "https://www.cancerimagingarchive.net/access-data/, then download the manifest 'Original ACRIN 6685 "
            "Images used to create Segmentations and Seed Points' from "
            "https://www.cancerimagingarchive.net/analysis-result/acrin-6685-tumor-annotations/ with the NBIA Data "
            f"Retriever and store the DICOM series in {image_dir}."
        )

    _preprocess_acrin_hnscc(annotation_dir, image_dir, metadata_path, preprocessed_dir, modality)
    return preprocessed_dir


def get_acrin_hnscc_paths(
    path: Union[os.PathLike, str], modality: Literal["PET", "CT"], download: bool = False
) -> List[str]:
    """Get paths to the ACRIN-HNSCC-FDG-PET-CT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The imaging modality. One of 'PET' or 'CT'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_acrin_hnscc_data(path, modality, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_acrin_hnscc_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Literal["PET", "CT"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ACRIN-HNSCC-FDG-PET-CT dataset for head and neck lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The imaging modality. One of 'PET' or 'CT'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_acrin_hnscc_paths(path, modality, download)

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


def get_acrin_hnscc_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Literal["PET", "CT"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ACRIN-HNSCC-FDG-PET-CT dataloader for head and neck lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The imaging modality. One of 'PET' or 'CT'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_acrin_hnscc_dataset(path, patch_shape, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
