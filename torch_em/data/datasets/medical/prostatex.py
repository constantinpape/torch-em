"""The PROSTATEx dataset contains annotations for prostate lesion and prostate zone segmentation
in multi-parametric prostate MRI.

The MRI scans are from the SPIE-AAPM-NCI PROSTATEx challenge (346 patients) and are distributed as DICOM series
on TCIA. The segmentation masks are from the third-party "PROSTATEx masks" repository (Cuocolo et al. 2021):
- Lesion masks for 299 lesions of 200 patients, on the axial T2-weighted and on the ADC images.
  The lesions are stored as instance labels, the instance id is the PROSTATEx finding id.
- Whole gland and zonal masks on the axial T2-weighted images for 204 patients.
  The zone labels are: 1: peripheral zone, 2: transition zone (the rest of the gland, i.e. transition zone,
  central zone and anterior fibromuscular stroma).

This module downloads only the annotated T2 and ADC series from TCIA, stacks them into volumes and stores them
together with the aligned masks in one hdf5 file per patient. The masks were drawn on NIfTI conversions of the DICOM
series (dcm2niix), so the module maps them back onto the DICOM grid and checks the alignment against the NIfTI images
shipped with the masks. The hdf5 groups are '<label_type>/<sequence>' with the datasets 'raw' and 'labels'
(e.g. 'lesions/t2/raw' and 'lesions/t2/labels'); the zones group additionally contains the binary
whole-gland mask ('zones/t2/prostate'). Note that for some patients the lesion and the zone masks were drawn on
different T2 series, hence the raw data is stored per group.

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/prostatex/
and the masks at https://github.com/rcuocolo/PROSTATEx_masks.

This dataset is from the publications https://doi.org/10.1109/TMI.2014.2303821 (PROSTATEx)
and https://doi.org/10.1016/j.ejrad.2021.109647 (masks).
The data was released at https://doi.org/10.7937/K9TCIA.2017.MURS5CL.
Please cite them if you use this dataset in your research.
"""

import os
import re
import csv
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from collections import defaultdict
from typing import Union, Tuple, List, Literal

import numpy as np
import requests

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


MASKS_COMMIT = "21b9dfde9da4f7b719c206fe1ca00ae31d6f5cf3"

URLS = {
    "images": f"{util.NBIA_API_URL}getSeries?Collection=PROSTATEx",
    "masks": f"https://github.com/rcuocolo/PROSTATEx_masks/archive/{MASKS_COMMIT}.zip",
}

CHECKSUMS = {
    "images": None,  # The DICOM series are downloaded individually from TCIA.
    "masks": None,  # GitHub does not guarantee stable archive checksums.
}

ZONE_IDS = {"peripheral_zone": 1, "transition_zone": 2}


def _load_dicom_volume(series_dir):
    """Stack a DICOM series into a volume with axes (z, y, x) and slices sorted along the slice normal."""
    import pydicom

    slices = [pydicom.dcmread(dcm_path) for dcm_path in natsorted(glob(os.path.join(series_dir, "*.dcm")))]
    slices = [dcm for dcm in slices if hasattr(dcm, "ImagePositionPatient")]
    orientation = np.array([float(v) for v in slices[0].ImageOrientationPatient])
    normal = np.cross(orientation[:3], orientation[3:])
    slices.sort(key=lambda dcm: np.dot(np.array([float(v) for v in dcm.ImagePositionPatient]), normal))
    return np.stack([dcm.pixel_array for dcm in slices])


def _nifti_to_dicom_grid(data):
    """Map a NIfTI array (x, y, z; dcm2niix conversion of an axial series) onto the DICOM grid (z, y, x).

    dcm2niix flips the row axis of axial series (DICOM rows run anterior -> posterior, the NIfTI y axis runs
    posterior -> anterior), so the array is transposed and flipped along y.
    """
    return np.asarray(data).transpose(2, 1, 0)[:, ::-1]


def _load_nifti_on_dicom_grid(path):
    import nibabel as nib

    nifti = nib.load(path)
    assert nib.aff2axcodes(nifti.affine) == ("L", "A", "S"), f"Unexpected axes for {path}"
    return _nifti_to_dicom_grid(nifti.dataobj)


def _parse_image_name(name):
    """Parse a name like 'ProstateX-0000_t2_tse_tra_4' into the patient id and the DICOM series number."""
    patient_id, _, series_number = re.match(r"(ProstateX-\d+)_(.*)_(\d+)$", name.strip()).groups()
    return patient_id, int(series_number)


def _read_image_lists(mask_dir):
    """Read the lists of the DICOM series the masks were drawn on.

    Returns a dict {(label_type, sequence, patient_id): (series_number, nifti_image_path)}.
    """
    image_series = {}
    with open(os.path.join(mask_dir, "lesions", "Image_list.csv"), "r") as f:
        for row in csv.DictReader(f):
            for sequence in ("T2", "ADC"):
                patient_id, series_number = _parse_image_name(row[sequence])
                image_path = os.path.join(mask_dir, "lesions", "Images", sequence, f"{row[sequence].strip()}.nii.gz")
                image_series[("lesions", sequence.lower(), patient_id)] = (series_number, image_path)
    with open(os.path.join(mask_dir, "prostate", "image_list.csv"), "r") as f:
        for row in csv.DictReader(f):
            patient_id, series_number = _parse_image_name(row["T2"])
            image_path = os.path.join(mask_dir, "prostate", "Images", f"{row['T2'].strip()}.nii.gz")
            image_series[("zones", "t2", patient_id)] = (series_number, image_path)
    return image_series


def _get_series_metadata(path, download):
    """Get the metadata of all series in the collection from the NBIA REST API."""
    metadata_path = os.path.join(path, "prostatex_series.json")
    if not os.path.exists(metadata_path):
        if not download:
            raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")
        response = requests.get(URLS["images"])
        response.raise_for_status()
        with open(metadata_path, "w") as f:
            json.dump(response.json(), f, indent=2)

    with open(metadata_path, "r") as f:
        return json.load(f)


def _find_mask(directory, patient_id, suffix=""):
    """Find the mask of a patient in one of the folders with the gland and zone masks.

    The file names are inconsistent in the mask repository (e.g. 'ProstateX-080.nii.gz' instead of
    'ProstateX-0080.nii.gz' and 'VOLUME-0202_pz.nii.gz' instead of 'ProstateX-0202_pz.nii.gz'),
    so the masks are matched via the patient number rather than via the file name.
    """
    number = int(patient_id.split("-")[-1])
    for mask_path in natsorted(glob(os.path.join(directory, f"*{suffix}.nii.gz"))):
        if int(re.search(r"(\d+)", os.path.basename(mask_path)).group(1)) == number:
            return mask_path
    return None


def _get_mask_paths(mask_dir, label_type, sequence, patient_id):
    """Get the mask files of a patient for the given label type and sequence."""
    if label_type == "lesions":
        return natsorted(glob(os.path.join(mask_dir, "lesions", "Masks", sequence.upper(), f"{patient_id}-*")))

    prostate_dir = os.path.join(mask_dir, "prostate")
    mask_paths = [
        _find_mask(os.path.join(prostate_dir, "mask_prostate"), patient_id),
        _find_mask(os.path.join(prostate_dir, "mask_pz"), patient_id, "_pz"),
        _find_mask(os.path.join(prostate_dir, "mask_tz"), patient_id, "_tz"),
    ]
    return mask_paths if all(mask_path is not None for mask_path in mask_paths) else []


def _find_series(dicom_dir, candidates, image_path, mask_shape):
    """Find the downloaded DICOM series that matches the NIfTI image the masks were drawn on.

    The NIfTI images that ship with the masks are used to identify the series: the stacked DICOM volume has to
    match the NIfTI image voxel by voxel. For the few patients without a NIfTI image in the mask repository the
    series is identified by the shape of the masks instead.

    Returns the volume on the DICOM grid or None if no candidate series matches.
    """
    reference = _load_nifti_on_dicom_grid(image_path) if os.path.exists(image_path) else None
    for series in candidates:
        series_dir = os.path.join(dicom_dir, series["SeriesInstanceUID"])
        if not os.path.exists(series_dir):
            continue
        volume = _load_dicom_volume(series_dir)
        if reference is None:
            if volume.shape == mask_shape:
                return volume
        elif volume.shape == reference.shape and np.array_equal(volume, reference.astype(volume.dtype)):
            return volume
    return None


def _load_lesion_labels(mask_paths, patient_id, shape):
    """Combine the per-lesion masks of a patient into an instance segmentation (id = finding id)."""
    labels = np.zeros(shape, dtype="uint8")
    pattern = re.compile(rf"{patient_id}-Finding(\d+)-.*ROI\.nii\.gz", re.IGNORECASE)
    finding_ids = [int(pattern.match(os.path.basename(mask_path)).group(1)) for mask_path in mask_paths]
    n_masks = 0
    for mask_path, finding_id in zip(mask_paths, finding_ids):
        # The finding ids start at 1, except for a single lesion in the current release of the masks
        # (ProstateX-0005), which is called 'Finding0' and gets the next free id instead of the background id.
        if finding_id == 0:
            finding_id = max(finding_ids) + 1
        mask = _load_nifti_on_dicom_grid(mask_path)
        if mask.shape != shape:  # A few masks were drawn on a different series and cannot be used.
            continue
        labels[mask > 0] = finding_id
        n_masks += 1
    return labels if n_masks > 0 else None


def _load_zone_labels(mask_paths, shape):
    """Combine the peripheral and transition zone masks into a semantic segmentation."""
    prostate, pz, tz = [_load_nifti_on_dicom_grid(mask_path) > 0 for mask_path in mask_paths]
    if any(mask.shape != shape for mask in (prostate, pz, tz)):
        return None, None
    zones = np.zeros(shape, dtype="uint8")
    zones[tz] = ZONE_IDS["transition_zone"]
    zones[pz] = ZONE_IDS["peripheral_zone"]
    return zones, prostate.astype("uint8")


def _preprocess_prostatex(dicom_dir, mask_dir, series_metadata, image_series, preprocessed_dir):
    import h5py

    series_by_number = defaultdict(list)
    for series in series_metadata:
        series_by_number[(series["PatientID"], int(series["SeriesNumber"]))].append(series)

    groups_per_patient = defaultdict(list)
    for (label_type, sequence, patient_id), (series_number, image_path) in image_series.items():
        groups_per_patient[patient_id].append((label_type, sequence, series_number, image_path))

    os.makedirs(preprocessed_dir, exist_ok=True)
    for patient_id, groups in tqdm(sorted(groups_per_patient.items()), desc="Preprocess PROSTATEx"):
        out_path = os.path.join(preprocessed_dir, f"{patient_id}.h5")
        if os.path.exists(out_path):
            continue

        datasets = {}
        for label_type, sequence, series_number, image_path in groups:
            mask_paths = _get_mask_paths(mask_dir, label_type, sequence, patient_id)
            if not mask_paths:  # Some patients only have masks for one of the label types.
                continue
            mask_shape = _load_nifti_on_dicom_grid(mask_paths[0]).shape
            candidates = series_by_number[(patient_id, series_number)]
            volume = _find_series(dicom_dir, candidates, image_path, mask_shape)
            if volume is None:
                raise RuntimeError(f"No DICOM series matches the masks of {patient_id} ({label_type}, {sequence}).")
            if label_type == "lesions":
                labels = _load_lesion_labels(mask_paths, patient_id, volume.shape)
                if labels is None:
                    continue
            else:
                labels, prostate = _load_zone_labels(mask_paths, volume.shape)
                if labels is None:
                    continue
                datasets[f"{label_type}/{sequence}/prostate"] = prostate
            datasets[f"{label_type}/{sequence}/raw"] = volume
            datasets[f"{label_type}/{sequence}/labels"] = labels

        if not datasets:
            continue
        tmp_path = out_path + ".tmp"
        with h5py.File(tmp_path, "w") as f:
            for key, data in datasets.items():
                f.create_dataset(key, data=data, compression="gzip")
        os.rename(tmp_path, out_path)


def get_prostatex_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PROSTATEx dataset.

    The download is resumable: series that were already downloaded and patients that were already converted are
    skipped when the function is called again.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    os.makedirs(path, exist_ok=True)
    preprocessed_dir = os.path.join(path, "preprocessed")

    # Download the masks.
    mask_dir = os.path.join(path, f"PROSTATEx_masks-{MASKS_COMMIT}", "Files")
    if not os.path.exists(mask_dir):
        zip_path = os.path.join(path, f"PROSTATEx_masks-{MASKS_COMMIT}.zip")
        util.download_source(path=zip_path, url=URLS["masks"], download=download, checksum=CHECKSUMS["masks"])
        util.unzip(zip_path=zip_path, dst=path)

    image_series = _read_image_lists(mask_dir)
    n_patients = len({patient_id for _, _, patient_id in image_series})
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == n_patients:
        return preprocessed_dir

    # Find the series the masks were drawn on and download them from TCIA.
    series_metadata = _get_series_metadata(path, download)
    series_numbers = {(patient_id, number) for (_, _, patient_id), (number, _) in image_series.items()}
    series_uids = sorted(
        series["SeriesInstanceUID"] for series in series_metadata
        if (series["PatientID"], int(series["SeriesNumber"])) in series_numbers
    )
    dicom_dir = os.path.join(path, "dicom")
    if download:  # Series that were downloaded already are skipped.
        util.download_tcia_series(series_uids, dst=dicom_dir, csv_filename=os.path.join(path, "prostatex_series"))
    elif not all(os.path.exists(os.path.join(dicom_dir, uid)) for uid in series_uids):
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    _preprocess_prostatex(dicom_dir, mask_dir, series_metadata, image_series, preprocessed_dir)
    return preprocessed_dir


def get_prostatex_paths(
    path: Union[os.PathLike, str],
    sequence: Literal["t2", "adc"] = "t2",
    label_type: Literal["lesions", "zones"] = "lesions",
    download: bool = False,
) -> List[str]:
    """Get paths to the PROSTATEx data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        sequence: The MRI sequence, either 't2' or 'adc'. The zone labels are only available for 't2'.
        label_type: The label type, either 'lesions' or 'zones'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('<label_type>/<sequence>/raw')
        and the label data ('<label_type>/<sequence>/labels').
    """
    import h5py

    assert sequence in ("t2", "adc"), f"Invalid sequence: {sequence}."
    assert label_type in ("lesions", "zones"), f"Invalid label type: {label_type}."
    if label_type == "zones" and sequence != "t2":
        raise ValueError("The zone labels are only available for the 't2' sequence.")

    data_dir = get_prostatex_data(path, download)
    volume_paths = []
    for volume_path in natsorted(glob(os.path.join(data_dir, "*.h5"))):
        with h5py.File(volume_path, "r") as f:
            if f"{label_type}/{sequence}/labels" in f:
                volume_paths.append(volume_path)
    return volume_paths


def get_prostatex_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    sequence: Literal["t2", "adc"] = "t2",
    label_type: Literal["lesions", "zones"] = "lesions",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PROSTATEx dataset for prostate lesion or zone segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        sequence: The MRI sequence, either 't2' or 'adc'. The zone labels are only available for 't2'.
        label_type: The label type, either 'lesions' or 'zones'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_prostatex_paths(path, sequence, label_type, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key=f"{label_type}/{sequence}/raw",
        label_paths=volume_paths,
        label_key=f"{label_type}/{sequence}/labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_prostatex_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    sequence: Literal["t2", "adc"] = "t2",
    label_type: Literal["lesions", "zones"] = "lesions",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PROSTATEx dataloader for prostate lesion or zone segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        sequence: The MRI sequence, either 't2' or 'adc'. The zone labels are only available for 't2'.
        label_type: The label type, either 'lesions' or 'zones'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_prostatex_dataset(path, patch_shape, sequence, label_type, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
