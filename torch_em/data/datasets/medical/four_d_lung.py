"""The 4D-Lung dataset contains longitudinal respiratory-correlated 4D fan-beam CT (4D-FBCT) and
4D cone-beam CT (4D-CBCT) acquisitions of 20 locally-advanced non-small cell lung cancer (NSCLC)
patients, collected over the course of chemoradiotherapy at the University of Texas MD Anderson
Cancer Center. Each patient contributes multiple 4D acquisitions (up to several hundred CT series
in total, ~589 studies for the full collection), of which a subset are paired with DICOM RTSTRUCT
gross tumor volume (GTV) contours drawn by radiation oncologists.

This loader only downloads the CT series that are referenced by an RTSTRUCT (rather than the full,
~183 GB collection), and rasterizes the GTV contours onto the CT grid with
`torch_em.data.datasets.util.rasterize_rtstruct`, storing image and label pairs as hdf5 files.
The labels are binary (foreground: GTV).

NOTE: This requires the pydicom python package.

The dataset is located at https://doi.org/10.7937/K9/TCIA.2016.ELN8YGLE and is fully public
(no data use agreement required), released under a CC-BY-3.0 license.

This dataset is from the publication https://doi.org/10.1002/mp.12059.
Please cite it if you use this dataset for your research.
"""

import os
import json
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

import requests
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


NBIA_API_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
COLLECTION = "4D-Lung"


def _get_series(path, modality):
    cache_path = os.path.join(path, f"series_{modality.lower()}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    response = requests.get(NBIA_API_URL + "getSeries", params={"Collection": COLLECTION, "Modality": modality})
    response.raise_for_status()
    series = response.json()

    os.makedirs(path, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(series, f)

    return series


def _referenced_series_uid(rtstruct_path):
    import pydicom

    rtstruct = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
    referenced_study = rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0]
    return str(referenced_study.RTReferencedSeriesSequence[0].SeriesInstanceUID)


def _preprocess_four_d_lung(rtstruct_series, rtstruct_dir, image_dir, preprocessed_dir):
    import h5py
    import numpy as np

    os.makedirs(preprocessed_dir, exist_ok=True)

    for series in tqdm(rtstruct_series, desc="Preprocess 4D-Lung"):
        uid = series["SeriesInstanceUID"]
        patient_id = series["PatientID"]
        out_path = os.path.join(preprocessed_dir, f"{patient_id}_{uid[-8:]}.h5")
        if os.path.exists(out_path):
            continue

        rtstruct_path = glob(os.path.join(rtstruct_dir, uid, "*.dcm"))[0]
        referenced_uid = _referenced_series_uid(rtstruct_path)
        image_series_dir = os.path.join(image_dir, referenced_uid)
        if not os.path.exists(image_series_dir):
            continue

        volume, geometry = util.load_dicom_series(image_series_dir)
        volume = np.round(volume).astype("int16")
        labels = util.rasterize_rtstruct(rtstruct_path, geometry, volume.shape, lambda roi_number, roi_name: 1)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_four_d_lung_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the 4D-Lung dataset.

    NOTE: This only downloads the CT series that have a matching RTSTRUCT (a small subset of the
    full, ~183 GB collection). Ensure enough disk space is available before running this with
    `download=True`, and expect the initial call to take a while, since every RTSTRUCT of the
    collection is inspected to find its referenced CT series.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if glob(os.path.join(preprocessed_dir, "*.h5")):
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)
    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    rtstruct_series = _get_series(path, "RTSTRUCT")
    rtstruct_dir = os.path.join(path, "rtstructs")
    util.download_tcia_series(
        [series["SeriesInstanceUID"] for series in rtstruct_series],
        dst=rtstruct_dir, csv_filename=os.path.join(path, "rtstructs"),
    )

    referenced_uids = sorted({
        _referenced_series_uid(glob(os.path.join(rtstruct_dir, series["SeriesInstanceUID"], "*.dcm"))[0])
        for series in rtstruct_series
    })
    image_dir = os.path.join(path, "images")
    util.download_tcia_series(referenced_uids, dst=image_dir, csv_filename=os.path.join(path, "images"))

    _preprocess_four_d_lung(rtstruct_series, rtstruct_dir, image_dir, preprocessed_dir)
    return preprocessed_dir


def get_four_d_lung_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the 4D-Lung data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_four_d_lung_data(path, download)
    return natsorted(glob(os.path.join(data_dir, "*.h5")))


def get_four_d_lung_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the 4D-Lung dataset for GTV segmentation in 4D lung CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_four_d_lung_paths(path, download)

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


def get_four_d_lung_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the 4D-Lung dataloader for GTV segmentation in 4D lung CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_four_d_lung_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
