"""The CC-Tumor-Heterogeneity (CCTH) dataset contains annotations for cervix and tumor segmentation in T2-weighted
MRI of patients with advanced cervical cancer.

It consists of 68 T2-weighted MRI volumes (67 sagittal, 1 axial) of 23 patients, acquired at up to three time
points during radiochemotherapy. Each volume comes with two contours drawn in MIM: the uterus / cervix (ROI with
magenta display color) and the tumor inside of it (ROI with cyan display color), which are distributed as DICOM
RTSTRUCT. This module downloads only the RTSTRUCT series and the T2-weighted MRI series they reference (not the
DCE / DWI MRI and PET/CT of the collection), rasterizes the contours onto the MRI grid
(see `torch_em.data.datasets.util.rasterize_rtstruct`) and stores images and labels in hdf5 files.
The semantic label ids are: 1: tumor, 2: cervix / uterus (excluding the tumor).

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/cc-tumor-heterogeneity/.

This dataset is from the publication https://doi.org/10.1016/j.ijrobp.2020.02.001.
The data was released at https://doi.org/10.7937/TCIA.2022.6X0F-2S2T.
Please cite it if you use this dataset in your research.
"""

import os
import csv
import requests
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


COLLECTION = "CC-Tumor-Heterogeneity"

# The DICOM series are downloaded individually from TCIA via the NBIA REST API.
URL = util.NBIA_API_URL + "getSeries"
CHECKSUM = None

LABEL_IDS = {"tumor": 1, "cervix": 2}

# The two ROIs of each RTSTRUCT have the same name and are only distinguished by their display color.
ROI_COLORS = {(0, 235, 235): LABEL_IDS["tumor"], (255, 0, 255): LABEL_IDS["cervix"]}


def _read_rtstruct_info(rtstruct_path):
    """Get the referenced series UID and the mapping from ROI numbers to label ids."""
    import pydicom

    rtstruct = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
    referenced_series = str(
        rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0]
        .RTReferencedSeriesSequence[0].SeriesInstanceUID
    )
    roi_labels = {}
    for roi_contour in rtstruct.ROIContourSequence:
        color = tuple(int(v) for v in roi_contour.ROIDisplayColor)
        assert color in ROI_COLORS, f"Unexpected ROI color {color} in {rtstruct_path}."
        roi_labels[int(roi_contour.ReferencedROINumber)] = ROI_COLORS[color]
    assert sorted(roi_labels.values()) == sorted(LABEL_IDS.values()), f"Unexpected ROIs in {rtstruct_path}."
    roi_names = {int(roi.ROINumber): str(roi.ROIName) for roi in rtstruct.StructureSetROISequence}
    return referenced_series, roi_labels, roi_names


def _preprocess_cc_tumor_heterogeneity(dicom_dir, csv_path, preprocessed_dir):
    import h5py

    with open(csv_path, "r") as f:
        rtstruct_series = {row["Series UID"]: row["Subject ID"] for row in csv.DictReader(f)}

    os.makedirs(preprocessed_dir, exist_ok=True)
    for series_uid, subject_id in tqdm(sorted(rtstruct_series.items()), desc="Preprocess CC-Tumor-Heterogeneity"):
        rtstruct_path = glob(os.path.join(dicom_dir, series_uid, "*.dcm"))[0]
        referenced_series, roi_labels, roi_names = _read_rtstruct_info(rtstruct_path)
        # The ROI names encode the imaging plane and time point, e.g. 'Ut-MRT2-Sag-1'.
        time_point = roi_names[1].replace("Ut-MRT2-", "")
        out_path = os.path.join(preprocessed_dir, f"{subject_id}_{time_point}.h5")
        if os.path.exists(out_path):
            continue

        volume, geometry = util.load_dicom_series(os.path.join(dicom_dir, referenced_series))
        labels = util.rasterize_rtstruct(
            rtstruct_path, geometry, volume.shape, lambda roi_number, roi_name: roi_labels[roi_number]
        )

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_cc_tumor_heterogeneity_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CC-Tumor-Heterogeneity dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    # NOTE: The preprocessing below skips volumes that were converted already, so an interrupted run resumes.
    preprocessed_dir = os.path.join(path, "preprocessed")
    dicom_dir = os.path.join(path, "dicom")
    if not os.path.exists(dicom_dir) and not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")
    os.makedirs(path, exist_ok=True)

    # Download the RTSTRUCT series of the collection, and then the T2-weighted MRI series they reference.
    # The series metadata are written after all series are downloaded, so their presence means it is complete.
    csv_path = os.path.join(path, "cc_tumor_heterogeneity_rtstruct.csv")
    image_csv_path = os.path.join(path, "cc_tumor_heterogeneity_images")
    if not os.path.exists(csv_path) or not os.path.exists(f"{image_csv_path}.csv"):
        response = requests.get(URL, params={"Collection": COLLECTION, "Modality": "RTSTRUCT"})
        response.raise_for_status()
        rtstruct_uids = sorted(series["SeriesInstanceUID"] for series in response.json())
        csv_path = util.download_tcia_series(rtstruct_uids, dicom_dir, csv_path[:-len(".csv")])

        image_uids = sorted(
            _read_rtstruct_info(glob(os.path.join(dicom_dir, uid, "*.dcm"))[0])[0] for uid in rtstruct_uids
        )
        util.download_tcia_series(image_uids, dicom_dir, image_csv_path)

    _preprocess_cc_tumor_heterogeneity(dicom_dir, csv_path, preprocessed_dir)
    return preprocessed_dir


def get_cc_tumor_heterogeneity_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the CC-Tumor-Heterogeneity data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_cc_tumor_heterogeneity_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_cc_tumor_heterogeneity_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CC-Tumor-Heterogeneity dataset for cervix and tumor segmentation in T2-weighted MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_cc_tumor_heterogeneity_paths(path, download)

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


def get_cc_tumor_heterogeneity_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CC-Tumor-Heterogeneity dataloader for cervix and tumor segmentation in T2-weighted MRI.

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
    dataset = get_cc_tumor_heterogeneity_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
