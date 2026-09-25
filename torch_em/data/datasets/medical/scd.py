"""The SCD dataset contains annotations for left ventricular endocardium and epicardium
segmentation in short-axis cardiac cine-MRI.

The Sunnybrook Cardiac Data (SCD), also known as the 2009 Cardiac MR Left Ventricle Segmentation
Challenge data, consists of 45 cine-MRI studies from a mix of patients and pathologies: healthy,
hypertrophy, heart failure with infarction and heart failure without infarction. Manual contours
of the left ventricular endocardium and epicardium are provided for the end-diastolic and
end-systolic frames (and, for a subset of slices, further frames), drawn by an expert
cardiologist. The dataset is located at https://www.cardiacatlas.org/sunnybrook-cardiac-data/
and is distributed under the CC0 1.0 Universal license.

The labels are multiclass, see `LABEL_IDS`: 1 = left ventricular cavity (endocardium), 2 = left
ventricular myocardium (the region between the epicardial and endocardial contour). Slices without
an epicardial contour only have the cavity label.

This dataset is from the publication http://hdl.handle.net/10380/3070.
Please cite it if you use this dataset for your research.
"""

import os
import re
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
from skimage.draw import polygon

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "images_1": "https://www.cardiacatlas.org/share/download.php?id=98&token=kUVtCTMzfjlHwunzFYST7yNGFnlUtV7W&download",  # noqa
    "images_2": "https://www.cardiacatlas.org/share/download.php?id=99&token=2euh4WH03DynHi9Y5elb2HSWJaJIeWER&download",  # noqa
    "images_3": "https://www.cardiacatlas.org/share/download.php?id=100&token=eT37AbXslu1JQp1GSZy2BmmWzWmXDdfx&download",  # noqa
    "images_4": "https://www.cardiacatlas.org/share/download.php?id=101&token=Qn0dY2lHTyTIBF5qNHRkbHdUSnWnS4Yy&download",  # noqa
    "images_5": "https://www.cardiacatlas.org/share/download.php?id=102&token=nYALJl6kS6cj5jgK9R8CKDpgQwZn1pm3&download",  # noqa
    "contours": "https://www.cardiacatlas.org/share/download.php?id=61&token=gFjv8psvCQu2vDYldUW9TYUIDic11fDt&download",  # noqa
    "patient_data": "https://www.cardiacatlas.org/share/download.php?id=66&token=Y1D66ieUdbXFlqmZ4icHJheJy44MXUPY&download",  # noqa
}

CHECKSUMS = {
    "images_1": "708ce04db1ac33948a00b9052d44e9548c6807121a4841f4c35080d6db127b72",
    "images_2": "ea97bd11dfc1154bdcf02b6466c00637bc7e0dc04b5ddc3d94d68a7cd7a3e6a5",
    "images_3": "5b065fadef1b523bbabfec0d85327f375f743f2dd72810a5927b21b834b45deb",
    "images_4": "2a8d7054d9b89b5af4c9764b4b04eecee57b61b71b69f5ac69ec12629e9a615c",
    "images_5": "e65a818fe89c665c344317222a052360e210936c81f89ceb9d36e8e3c6e2d6e9",
    "contours": "93d8e06dfa53fc384e78810aeac86c336b5bdecbff1c9c175ec9254851cc2162",
    "patient_data": "c39eb11924d021e8ec51bb984daa57d1adadcc8eb7e342640181ce151dae5ded",
}

LABEL_IDS = {"background": 0, "cavity": 1, "myocardium": 2}


def _normalize_original_id(original_id):
    """Zero-pad the trailing case number of an 'OriginalID' to match the contour directory naming,
    e.g. 'SC-HF-I-1' -> 'SC-HF-I-01'."""
    return re.sub(r"-(\d+)$", lambda m: f"-{int(m.group(1)):02d}", original_id)


def _patient_id_mapping(patient_csv_path):
    import csv

    mapping = {}
    with open(patient_csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mapping[_normalize_original_id(row["OriginalID"])] = row["PatientID"]
    return mapping


def _parse_contour(contour_path):
    points = np.loadtxt(contour_path)
    return points[:, 0], points[:, 1]


INSTANCE_PATTERN = re.compile(r"IM-\d+-(\d+)-[io]contour-manual\.txt$")


def _index_cinesax_instances(patient_dir):
    """Index the DICOM instances of the short-axis cine series ('CINESAX') of a patient folder by their
    instance number. The Cardiac Atlas Project redistribution stores several series per patient (long-axis
    cines, scouts, perfusion, ...) under arbitrarily numbered subfolders, and only the DICOM header's
    'SeriesDescription' reliably identifies the short-axis cine series that the manual contours refer to;
    the numeric prefix of the contour filenames (e.g. 'IM-0001-0048') does not correspond to the subfolder
    or series naming of this redistribution."""
    import pydicom

    instances = {}
    for series_dir in sorted(p for p in glob(os.path.join(patient_dir, "*")) if os.path.isdir(p)):
        dicom_paths = natsorted(glob(os.path.join(series_dir, "*.dcm")))
        if not dicom_paths:
            continue

        header = pydicom.dcmread(dicom_paths[0], stop_before_pixels=True)
        if "CINESAX" not in getattr(header, "SeriesDescription", "").upper():
            continue

        for dicom_path in dicom_paths:
            match = re.search(r"-(\d+)\.dcm$", os.path.basename(dicom_path))
            if match is None:
                continue
            instances.setdefault(int(match.group(1)), dicom_path)

    return instances


def _rasterize_case(instances, contour_dir, preprocessed_dir, patient_id):
    import pydicom
    import imageio.v3 as imageio

    icontours = natsorted(glob(os.path.join(contour_dir, "*-icontour-manual.txt")))

    for icontour_path in icontours:
        match = INSTANCE_PATTERN.search(os.path.basename(icontour_path))
        if match is None or int(match.group(1)) not in instances:
            continue

        img_no = int(match.group(1))
        stem = f"{img_no:04}"
        gt_path = os.path.join(preprocessed_dir, f"{patient_id}_{stem}.tif")
        raw_path = os.path.join(preprocessed_dir, f"{patient_id}_{stem}_raw.tif")
        if os.path.exists(gt_path) and os.path.exists(raw_path):
            continue

        dcm = pydicom.dcmread(instances[img_no])
        image = np.asarray(dcm.pixel_array)
        shape = image.shape

        labels = np.zeros(shape, dtype="uint8")

        ocontour_path = os.path.join(contour_dir, os.path.basename(icontour_path).replace("icontour", "ocontour"))
        if os.path.exists(ocontour_path):
            x, y = _parse_contour(ocontour_path)
            r, c = polygon(y, x, shape=shape)
            labels[r, c] = LABEL_IDS["myocardium"]

        x, y = _parse_contour(icontour_path)
        r, c = polygon(y, x, shape=shape)
        labels[r, c] = LABEL_IDS["cavity"]

        imageio.imwrite(raw_path, image)
        imageio.imwrite(gt_path, labels)


def _preprocess_inputs(path, preprocessed_dir):
    os.makedirs(preprocessed_dir, exist_ok=True)

    patient_csv_path = os.path.join(path, "scd_patientdata.csv")
    mapping = _patient_id_mapping(patient_csv_path)

    contours_root = os.path.join(path, "SCD_ManualContours")
    case_dirs = natsorted(glob(os.path.join(contours_root, "SC-*")))

    for case_dir in tqdm(case_dirs, desc="Preprocessing the SCD studies"):
        original_id = os.path.basename(case_dir)
        patient_id = mapping.get(original_id)
        if patient_id is None:
            continue

        contour_dir = os.path.join(case_dir, "contours-manual", "IRCCI-expert")
        if not os.path.exists(contour_dir):
            continue

        patient_dir = os.path.join(path, patient_id)
        instances = _index_cinesax_instances(patient_dir)
        if not instances:
            continue

        _rasterize_case(instances, contour_dir, preprocessed_dir, patient_id)


def get_scd_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the SCD dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if glob(os.path.join(preprocessed_dir, "*_raw.tif")):
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    for key in ["images_1", "images_2", "images_3", "images_4", "images_5"]:
        zip_path = os.path.join(path, f"{key}.zip")
        util.download_source(
            path=zip_path, url=URLS[key], download=download, checksum=CHECKSUMS[key], verify=False
        )
        util.unzip(zip_path=zip_path, dst=path)

    contours_dir = os.path.join(path, "SCD_ManualContours")
    if not os.path.exists(contours_dir):
        zip_path = os.path.join(path, "contours.zip")
        util.download_source(
            path=zip_path, url=URLS["contours"], download=download, checksum=CHECKSUMS["contours"], verify=False
        )
        util.unzip(zip_path=zip_path, dst=path)

    patient_csv_path = os.path.join(path, "scd_patientdata.csv")
    util.download_source(
        path=patient_csv_path,
        url=URLS["patient_data"],
        download=download,
        checksum=CHECKSUMS["patient_data"],
        verify=False,
    )

    _preprocess_inputs(path, preprocessed_dir)
    return preprocessed_dir


def get_scd_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the SCD data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_scd_data(path, download)

    gt_paths = natsorted(glob(os.path.join(data_dir, "*.tif")))
    gt_paths = [p for p in gt_paths if not p.endswith("_raw.tif")]
    image_paths = [p.replace(".tif", "_raw.tif") for p in gt_paths]

    return image_paths, gt_paths


def get_scd_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SCD dataset for left ventricular cavity and myocardium segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_scd_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_scd_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SCD dataloader for left ventricular cavity and myocardium segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_scd_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
