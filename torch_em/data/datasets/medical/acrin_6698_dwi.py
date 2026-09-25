"""The ACRIN-6698 dataset contains annotations for the whole breast tumor in diffusion-weighted MRI.

This module covers the 1103 'DWI SEG' objects of the ACRIN-6698 / I-SPY2 collection: manual whole-tumor
segmentations on the apparent diffusion coefficient (ADC) map of breast cancer patients undergoing
neoadjuvant chemotherapy. Each DICOM-SEG object is named '...DWI SEG: from S<n>: Whole Tumor Manual',
where '<n>' is the DICOM series number of the ADC map it was drawn on; this module resolves that
reference by series number rather than a DICOM reference, since the SEG objects in this collection
carry none (neither a `ReferencedSeriesSequence` nor a per-frame derivation link).

NOTE: The collection's other, unrelated family of 1110 'ISPY2: VOLSER' functional-tumor-volume masks
(thresholded from a signal-enhancement-ratio map, not a manual DWI segmentation) is covered separately
by `medical.acrin_6698`.

NOTE: This requires the pydicom python package.

The dataset is located at https://doi.org/10.7937/TCIA.kk02-6d95 and is distributed under the
CC BY 4.0 license.
Please cite it if you use this dataset in your research.
"""

import os
import re
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .adrenal_acc import _load_dicom_volume, _load_dicom_seg, _resample_labels
from .. import util


COLLECTION = "ACRIN-6698"


def _get_series_metadata(path, download):
    """Get the metadata of all series in the collection from the NBIA REST API."""
    import requests

    metadata_path = os.path.join(path, "acrin_6698_dwi_series.json")
    if not os.path.exists(metadata_path):
        if not download:
            raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")
        response = requests.get(f"{util.NBIA_API_URL}getSeries", params={"Collection": COLLECTION})
        response.raise_for_status()
        with open(metadata_path, "w") as f:
            json.dump(response.json(), f, indent=2)

    with open(metadata_path, "r") as f:
        return json.load(f)


def _is_dwi_seg(series):
    return series.get("Modality") == "SEG" and "DWI SEG" in (series.get("SeriesDescription") or "")


def _referenced_series_number(series):
    match = re.search(r"from S(\d+)", series["SeriesDescription"])
    return match.group(1) if match else None


def _find_adc_series(series_metadata, study_uid, series_number):
    """Find the ADC map a DWI SEG was drawn on, matched by the series number in its own name."""
    candidates = [
        series for series in series_metadata
        if series.get("StudyInstanceUID") == study_uid and series.get("Modality") == "MR"
        and f"ADC: from S{series_number}:" in (series.get("SeriesDescription") or "")
    ]
    return candidates[0]["SeriesInstanceUID"] if len(candidates) == 1 else None


def _preprocess_acrin_6698_dwi(dicom_dir, series_metadata, preprocessed_dir):
    import h5py

    seg_series = [series for series in series_metadata if _is_dwi_seg(series)]

    os.makedirs(preprocessed_dir, exist_ok=True)
    for series in tqdm(seg_series, desc="Preprocess ACRIN-6698 (DWI)"):
        seg_paths = glob(os.path.join(dicom_dir, series["SeriesInstanceUID"], "*.dcm"))
        if not seg_paths:
            continue

        out_path = os.path.join(preprocessed_dir, f"{series['SeriesInstanceUID']}.h5")
        if os.path.exists(out_path):
            continue

        series_number = _referenced_series_number(series)
        if series_number is None:
            continue
        adc_uid = _find_adc_series(series_metadata, series["StudyInstanceUID"], series_number)
        adc_dir = os.path.join(dicom_dir, adc_uid) if adc_uid else None
        if adc_dir is None or not glob(os.path.join(adc_dir, "*.dcm")):
            continue

        volume, adc_affine = _load_dicom_volume(adc_dir)
        seg_labels, seg_affine = _load_dicom_seg(seg_paths[0])
        # The SEG only covers the slices where the tumor is present, cropped from the full ADC extent,
        # so it is placed on the ADC's own grid rather than compared to it directly.
        labels = _resample_labels(seg_labels, seg_affine, volume.shape, adc_affine)
        if labels.max() == 0:
            continue

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_acrin_6698_dwi_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ACRIN-6698 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    # NOTE: The preprocessing below skips volumes that were converted already, so an interrupted run resumes.
    preprocessed_dir = os.path.join(path, "preprocessed")

    os.makedirs(path, exist_ok=True)
    series_metadata = _get_series_metadata(path, download)

    seg_uids = [series["SeriesInstanceUID"] for series in series_metadata if _is_dwi_seg(series)]

    dicom_dir = os.path.join(path, "dicom")
    if download:  # The SEG series are downloaded first, so the ADC series they reference can be found.
        util.download_tcia_series(seg_uids, dst=dicom_dir, csv_filename=os.path.join(path, "acrin_6698_dwi_seg"))

        seg_by_uid = {series["SeriesInstanceUID"]: series for series in series_metadata if _is_dwi_seg(series)}
        adc_uids = set()
        for uid in seg_uids:
            series = seg_by_uid[uid]
            series_number = _referenced_series_number(series)
            if series_number is None:
                continue
            adc_uid = _find_adc_series(series_metadata, series["StudyInstanceUID"], series_number)
            if adc_uid:
                adc_uids.add(adc_uid)
        util.download_tcia_series(
            sorted(adc_uids), dst=dicom_dir, csv_filename=os.path.join(path, "acrin_6698_dwi_adc")
        )
    elif not glob(os.path.join(dicom_dir, "*", "*.dcm")):
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    _preprocess_acrin_6698_dwi(dicom_dir, series_metadata, preprocessed_dir)
    return preprocessed_dir


def get_acrin_6698_dwi_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the ACRIN-6698 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    preprocessed_dir = get_acrin_6698_dwi_data(path, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{preprocessed_dir}'."
    return volume_paths


def get_acrin_6698_dwi_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ACRIN-6698 dataset for breast tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_acrin_6698_dwi_paths(path, download)

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


def get_acrin_6698_dwi_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ACRIN-6698 dataloader for breast tumor segmentation.

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
    dataset = get_acrin_6698_dwi_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
