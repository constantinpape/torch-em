"""The LIDC-IDRI dataset contains annotations for lung nodule segmentation in thoracic CT.

It consists of 1018 CT scans of 1010 patients, in which up to four thoracic radiologists outlined the nodules
with a diameter of at least 3 mm. The CT scans are distributed as DICOM series (the CR / DX scans of the collection
are skipped) and the annotations as XML files with per-radiologist nodule contours. This module stacks the DICOM
series into volumes, rasterizes the contours onto the CT grid and stores them together in hdf5 files.

The contours of a radiologist are rasterized per slice with `skimage.draw.polygon`. Following the LIDC convention
(and pylidc), the contour points themselves are not part of the nodule and 'exclusion' contours are subtracted.
The individual radiologist annotations are grouped into nodules by voxel overlap: annotations of different
radiologists that overlap are treated as the same nodule. Each nodule gets one instance id and the hdf5 files
store the nodule masks for all consensus levels ('labels/consensus_1' to 'labels/consensus_4'), where consensus
level n contains the voxels that at least n radiologists marked as part of the nodule (level 1 is the union,
level 4 is the intersection of all four readings). The number of radiologists that annotated each voxel is stored
in 'labels/n_readers'. Nodules < 3 mm and non-nodules (single point marks) are not rasterized.

NOTE: This requires the pydicom python package.

The dataset is located at https://www.cancerimagingarchive.net/collection/lidc-idri/.

This dataset is from the publication https://doi.org/10.1118/1.3528204.
The data was released at https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX.
Please cite it if you use this dataset in your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from collections import defaultdict
from typing import Union, Tuple, List
import xml.etree.ElementTree as ET

import numpy as np
import requests

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "images": f"{util.NBIA_API_URL}getSeries?Collection=LIDC-IDRI&Modality=CT",
    "annotations": "https://www.cancerimagingarchive.net/wp-content/uploads/LIDC-XML-only.zip",
}

CHECKSUMS = {
    "images": None,  # The DICOM series are downloaded individually from TCIA.
    "annotations": "644557a3aa305602609c718b0cae33a93be762e8901a80bcd9df735b0ec6ab90",
}

XML_NAMESPACE = "{http://www.nih.gov}"


def _load_dicom_volume(series_dir):
    """Stack a DICOM series into a volume with axes (z, y, x) and slices sorted by ascending patient z position.

    Returns the volume in Hounsfield units, the SOP instance UID and the z position of each slice.
    """
    import pydicom

    slices = [pydicom.dcmread(dcm_path) for dcm_path in natsorted(glob(os.path.join(series_dir, "*.dcm")))]
    slices = [dcm for dcm in slices if hasattr(dcm, "ImagePositionPatient")]
    slices.sort(key=lambda dcm: float(dcm.ImagePositionPatient[2]))

    volume = np.stack([dcm.pixel_array for dcm in slices]).astype("float32")
    volume = volume * float(slices[0].RescaleSlope) + float(slices[0].RescaleIntercept)
    volume = np.round(volume).astype("int16")

    sop_uids = [str(dcm.SOPInstanceUID) for dcm in slices]
    z_positions = np.array([float(dcm.ImagePositionPatient[2]) for dcm in slices])
    return volume, sop_uids, z_positions


def _parse_annotations(xml_path):
    """Parse the nodule (>= 3 mm) contours of all reading sessions in a LIDC XML file.

    Returns a list with one entry per radiologist (reading session). Each entry is a list of nodules and each nodule
    is a list of contours (imageSOP_UID, imageZposition, inclusion, points), where points is an array of
    (row, column) coordinates. Nodules < 3 mm (a single point mark) are skipped.
    """
    root = ET.parse(xml_path).getroot()
    sessions = []
    for session in root.iter(f"{XML_NAMESPACE}readingSession"):
        nodules = []
        for nodule in session.iter(f"{XML_NAMESPACE}unblindedReadNodule"):
            contours = []
            for roi in nodule.iter(f"{XML_NAMESPACE}roi"):
                points = np.array([
                    [int(edge.find(f"{XML_NAMESPACE}yCoord").text), int(edge.find(f"{XML_NAMESPACE}xCoord").text)]
                    for edge in roi.iter(f"{XML_NAMESPACE}edgeMap")
                ])
                inclusion = roi.find(f"{XML_NAMESPACE}inclusion").text.strip().upper() == "TRUE"
                sop_uid = roi.find(f"{XML_NAMESPACE}imageSOP_UID").text.strip()
                z_position = float(roi.find(f"{XML_NAMESPACE}imageZposition").text)
                contours.append((sop_uid, z_position, inclusion, points))
            # Nodules < 3 mm are marked with a single point and are not segmented.
            if all(len(points) < 3 for _, _, _, points in contours):
                continue
            nodules.append(contours)
        sessions.append(nodules)
    return sessions


def _find_slice(sop_uid, z_position, slice_ids, z_positions):
    """Find the slice a contour belongs to via its SOP instance UID.

    For a few scans the XML files reference SOP instance UIDs that do not exist in the DICOM series
    (e.g. LIDC-IDRI-0017). In this case the slice is found via the z position, which has to match a slice
    position exactly.
    """
    if sop_uid in slice_ids:
        return slice_ids[sop_uid]
    z = int(np.argmin(np.abs(z_positions - z_position)))
    if abs(z_positions[z] - z_position) > 1e-3:
        raise ValueError(f"Cannot find the slice for the SOP instance UID {sop_uid} at z position {z_position}.")
    return z


def _rasterize_nodule(contours, shape, slice_ids, z_positions):
    """Rasterize the contours of one nodule annotation onto the CT grid."""
    from skimage.draw import polygon

    mask = np.zeros(shape, dtype="bool")
    # Fill the inclusion contours first, then subtract the exclusion contours.
    for inclusion in (True, False):
        for sop_uid, z_position, is_inclusion, points in contours:
            if is_inclusion != inclusion or len(points) < 3:
                continue
            z = _find_slice(sop_uid, z_position, slice_ids, z_positions)
            rr, cc = polygon(points[:, 0], points[:, 1], shape=shape[1:])
            if inclusion:
                mask[z, rr, cc] = True
            else:
                mask[z, rr, cc] = False
            # The contour points are not part of the nodule (they lie just outside of it).
            mask[z, points[:, 0], points[:, 1]] = False
    return mask


def _build_nodule_labels(sessions, shape, slice_ids, z_positions):
    """Group the per-radiologist nodule annotations into nodules and derive the consensus labels.

    Returns the instance labels for the consensus levels 1 to 4 and the number of readers per voxel.
    """
    # Rasterize all annotations and keep only the crop around each annotation to save memory.
    annotations = []
    for reader_id, nodules in enumerate(sessions):
        for contours in nodules:
            mask = _rasterize_nodule(contours, shape, slice_ids, z_positions)
            if not mask.any():
                continue
            bbox = tuple(slice(int(c.min()), int(c.max()) + 1) for c in np.where(mask))
            annotations.append((reader_id, bbox, mask[bbox]))

    def intersect(bbox_a, bbox_b):
        bbox = tuple(slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(bbox_a, bbox_b))
        return bbox if all(b.stop > b.start for b in bbox) else None

    def crop(mask, bbox, sub_bbox):
        return mask[tuple(slice(s.start - b.start, s.stop - b.start) for s, b in zip(sub_bbox, bbox))]

    # Group the annotations by voxel overlap (union-find over the pairwise overlaps).
    parents = list(range(len(annotations)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i in range(len(annotations)):
        for j in range(i + 1, len(annotations)):
            _, bbox_i, mask_i = annotations[i]
            _, bbox_j, mask_j = annotations[j]
            overlap = intersect(bbox_i, bbox_j)
            if overlap is not None and np.any(crop(mask_i, bbox_i, overlap) & crop(mask_j, bbox_j, overlap)):
                parents[find(i)] = find(j)

    groups = defaultdict(list)
    for i in range(len(annotations)):
        groups[find(i)].append(i)

    n_readers = np.zeros(shape, dtype="uint8")
    consensus = {level: np.zeros(shape, dtype="uint16") for level in range(1, 5)}
    for nodule_id, members in enumerate(groups.values(), start=1):
        # Count the readers per voxel (a reader may have annotated the same nodule more than once).
        counts = np.zeros(shape, dtype="uint8")
        for reader_id in set(annotations[i][0] for i in members):
            reader_mask = np.zeros(shape, dtype="bool")
            for i in members:
                if annotations[i][0] == reader_id:
                    reader_mask[annotations[i][1]] |= annotations[i][2]
            counts += reader_mask
        n_readers = np.maximum(n_readers, counts)
        for level in consensus:
            consensus[level][counts >= level] = nodule_id

    return consensus, n_readers


def _preprocess_lidc_idri(dicom_dir, xml_dir, series_metadata, preprocessed_dir):
    import h5py

    # Map the series UIDs to the XML files. A few series have multiple (identical) XML files
    # and the file for LIDC-IDRI-0101 was resubmitted with a correction, which takes precedence.
    xml_paths = {}
    all_xml_paths = glob(os.path.join(xml_dir, "**", "*.xml"), recursive=True)
    all_xml_paths += glob(os.path.join(xml_dir, "..", "*.xml"))
    for xml_path in natsorted(all_xml_paths):
        with open(xml_path, "r", errors="replace") as f:
            header = f.read(4096)
        if "LidcReadMessage" not in header:  # Skip the CXR annotations.
            continue
        start = header.index("<SeriesInstanceUid>") + len("<SeriesInstanceUid>")
        series_uid = header[start:header.index("</SeriesInstanceUid>")].strip()
        if series_uid not in xml_paths or "correction" in os.path.basename(xml_path):
            xml_paths[series_uid] = xml_path

    # The 8 patients with two CT series get one file per series (suffixed with the series number).
    series_per_patient = defaultdict(list)
    for series in series_metadata:
        series_per_patient[series["PatientID"]].append(series)

    os.makedirs(preprocessed_dir, exist_ok=True)
    for series in tqdm(series_metadata, desc="Preprocess LIDC-IDRI"):
        patient_id, series_uid = series["PatientID"], series["SeriesInstanceUID"]
        name = patient_id if len(series_per_patient[patient_id]) == 1 else f"{patient_id}_{series['SeriesNumber']}"
        out_path = os.path.join(preprocessed_dir, f"{name}.h5")
        if os.path.exists(out_path):
            continue

        series_dir = os.path.join(dicom_dir, series_uid)
        if not os.path.exists(series_dir):
            continue

        volume, sop_uids, z_positions = _load_dicom_volume(series_dir)
        slice_ids = {sop_uid: z for z, sop_uid in enumerate(sop_uids)}
        sessions = _parse_annotations(xml_paths[series_uid])
        consensus, n_readers = _build_nodule_labels(sessions, volume.shape, slice_ids, z_positions)

        tmp_path = out_path + ".tmp"
        with h5py.File(tmp_path, "w") as f:
            f.attrs["patient_id"] = patient_id
            f.attrs["series_uid"] = series_uid
            f.attrs["n_readers"] = len(sessions)
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels/n_readers", data=n_readers, compression="gzip")
            for level, labels in consensus.items():
                f.create_dataset(f"labels/consensus_{level}", data=labels, compression="gzip")
        os.rename(tmp_path, out_path)


def _get_series_metadata(path, download):
    """Get the metadata of all CT series in the collection from the NBIA REST API."""
    metadata_path = os.path.join(path, "lidc_idri_ct_series.json")
    if not os.path.exists(metadata_path):
        if not download:
            raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")
        response = requests.get(URLS["images"])
        response.raise_for_status()
        with open(metadata_path, "w") as f:
            json.dump(response.json(), f, indent=2)

    with open(metadata_path, "r") as f:
        series_metadata = json.load(f)
    return sorted(series_metadata, key=lambda series: (series["PatientID"], series["SeriesInstanceUID"]))


def get_lidc_idri_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the LIDC-IDRI dataset.

    The download is resumable: series that were already downloaded and volumes that were already converted are
    skipped when the function is called again.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    os.makedirs(path, exist_ok=True)
    preprocessed_dir = os.path.join(path, "preprocessed")

    # The list of CT series (the CR / DX scans in the collection are skipped).
    series_metadata = _get_series_metadata(path, download)
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == len(series_metadata):
        return preprocessed_dir

    # Download the XML annotations.
    xml_dir = os.path.join(path, "tcia-lidc-xml")
    if not os.path.exists(xml_dir):
        zip_path = os.path.join(path, "LIDC-XML-only.zip")
        util.download_source(
            path=zip_path, url=URLS["annotations"], download=download, checksum=CHECKSUMS["annotations"]
        )
        util.unzip(zip_path=zip_path, dst=path)

    # Download the CT series from TCIA (series that were downloaded already are skipped).
    dicom_dir = os.path.join(path, "dicom")
    series_uids = [series["SeriesInstanceUID"] for series in series_metadata]
    if download:
        util.download_tcia_series(series_uids, dst=dicom_dir, csv_filename=os.path.join(path, "lidc_idri_series"))
    elif not all(os.path.exists(os.path.join(dicom_dir, uid)) for uid in series_uids):
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    _preprocess_lidc_idri(dicom_dir, xml_dir, series_metadata, preprocessed_dir)
    return preprocessed_dir


def get_lidc_idri_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the LIDC-IDRI data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data
        ('labels/consensus_<level>' and 'labels/n_readers').
    """
    data_dir = get_lidc_idri_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_lidc_idri_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    consensus_level: int = 1,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LIDC-IDRI dataset for lung nodule segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        consensus_level: The minimum number of radiologists (1 to 4) that have to agree on a voxel for it to be part
            of a nodule. 1 corresponds to the union and 4 to the intersection of all radiologist annotations.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert consensus_level in (1, 2, 3, 4), f"Invalid consensus level: {consensus_level}."
    volume_paths = get_lidc_idri_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=f"labels/consensus_{consensus_level}",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_lidc_idri_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    consensus_level: int = 1,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LIDC-IDRI dataloader for lung nodule segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        consensus_level: The minimum number of radiologists (1 to 4) that have to agree on a voxel for it to be part
            of a nodule. 1 corresponds to the union and 4 to the intersection of all radiologist annotations.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lidc_idri_dataset(path, patch_shape, consensus_level, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
