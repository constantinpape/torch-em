"""The SAROS dataset contains annotations for 13 body regions and 6 body parts in whole-body CT.

The dataset consists of 900 CT series pooled from 28 TCIA collections, each resampled to 5mm slice
thickness and given two label volumes on that same grid: `body-regions.nii.gz` (see `BODY_REGIONS`)
and `body-parts.nii.gz` (see `BODY_PARTS`). Both are sparsely annotated: only every 5th axial slice
was reviewed by an annotator, and `IGNORE_LABEL` marks every other slice.

NOTE: The images are not distributed with the release: only the label volumes and a manifest CSV
are, so this module downloads and reconstructs them from their original TCIA series, following the
same steps and settings as the release. A raw DICOM conversion (e.g. with dcm2niix) does not share
the label's grid, so it is resampled onto it: DICOM patient coordinates are LPS, the label is stored
in a RAS+ world frame, and a trilinear resampling with a -1024 HU fill value outside the CT extent
completes the match.

NOTE: 6 of the 28 source collections (Head-Neck Cetuximab, ACRIN-HNSCC-FDG-PET-CT, QIN-HEADNECK,
TCGA-HNSC, HNSCC, Anti-PD-1_MELANOMA) require signing a TCIA Restricted License Agreement and are
not reachable through the public NBIA API, so their 174 cases are skipped; the remaining 726 are
openly downloadable.

NOTE: This requires the pydicom, nibabel and scipy python packages.

The dataset is located at https://doi.org/10.25737/sz96-zg60 and is distributed under the
TCIA Restricted License / CC BY 4.0 license (per-collection, see the collection's own citation).
This dataset is from the publication https://doi.org/10.1038/s41597-024-03337-6.
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

from .adrenal_acc import _load_dicom_volume
from .. import util


URLS = {
    "segs": "https://www.cancerimagingarchive.net/wp-content/uploads/SAROS-Collection-NIfTI-files-v2_03-70-2024.zip",  # noqa
    "info": "https://www.cancerimagingarchive.net/wp-content/uploads/Segmentation-Info_09-29-2023.csv",
}

CHECKSUMS = {
    "segs": "b509ff70fa69673b0697dac711a92b0e04476780feadf089927a7c8fcd7037e5",
    "info": "dac6df664279965567b79ff816a23d6f851cd7ab23340e81ff581c9b079c0cb1",
}

RESTRICTED_COLLECTIONS = {
    "Head-Neck Cetuximab", "ACRIN-HNSCC-FDG-PET-CT", "QIN-HEADNECK", "TCGA-HNSC", "HNSCC", "Anti-PD-1_MELANOMA",
}
"""The source collections that require a TCIA Restricted License Agreement and are skipped."""

IGNORE_LABEL = 255
"""The sentinel that marks a voxel outside the sparsely reviewed slices."""

BODY_REGIONS = {
    "subcutaneous_tissue": 1, "muscle": 2, "abdominal_cavity": 3, "thoracic_cavity": 4, "bone": 5,
    "parotid_glands": 6, "pericardium": 7, "breast_implant": 8, "mediastinum": 9, "brain": 10,
    "spinal_cord": 11, "thyroid_glands": 12, "submandibular_glands": 13,
}
"""Mapping from the body region name to its label id in `body-regions.nii.gz`."""

BODY_PARTS = {"torso": 1, "head": 2, "right_leg": 3, "left_leg": 4, "right_arm": 5, "left_arm": 6}
"""Mapping from the body part name to its label id in `body-parts.nii.gz`."""


def _read_manifest(info_path):
    with open(info_path) as f:
        return list(csv.DictReader(f))


def _resample_to_label(volume, ct_affine, label_shape, label_affine):
    """Resample a DICOM-derived volume onto the grid of its label, matching the release's own
    reconstruction: DICOM patient coordinates are LPS, converted to the RAS+ frame of the label by
    negating x and y, then a trilinear resampling with a -1024 HU fill value outside the CT extent.
    """
    from scipy.ndimage import affine_transform

    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    ras_affine = lps_to_ras @ ct_affine
    if volume.shape == label_shape and np.allclose(ras_affine, label_affine, atol=1e-3):
        return volume.astype("int16")

    to_ct_index = np.linalg.inv(ras_affine) @ label_affine
    resampled = affine_transform(
        volume.astype("float32"), to_ct_index[:3, :3], offset=to_ct_index[:3, 3],
        output_shape=label_shape, order=1, mode="constant", cval=-1024.0,
    )
    return np.round(resampled).astype("int16")


def _preprocess_saros(seg_dir, manifest, dicom_dir, preprocessed_dir):
    import h5py
    import nibabel as nib

    os.makedirs(preprocessed_dir, exist_ok=True)
    for row in tqdm(manifest, desc="Preprocess SAROS"):
        case_id = row["id"]
        out_path = os.path.join(preprocessed_dir, f"{case_id}.h5")
        if os.path.exists(out_path):
            continue

        regions_path = os.path.join(seg_dir, case_id, "body-regions.nii.gz")
        parts_path = os.path.join(seg_dir, case_id, "body-parts.nii.gz")
        if not (os.path.exists(regions_path) and os.path.exists(parts_path)):
            continue

        series_dir = os.path.join(dicom_dir, row["tcia_series_instance_uid"])
        if not glob(os.path.join(series_dir, "*.dcm")):
            continue

        regions_image = nib.load(regions_path)
        regions = np.asarray(regions_image.dataobj)
        parts = np.asarray(nib.load(parts_path).dataobj)

        volume, ct_affine = _load_dicom_volume(series_dir)
        raw = _resample_to_label(volume, ct_affine, regions.shape, regions_image.affine)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels/regions", data=regions.astype("uint8"), compression="gzip")
            f.create_dataset("labels/parts", data=parts.astype("uint8"), compression="gzip")


def get_saros_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the SAROS dataset.

    The images are reconstructed from TCIA, which is several hundred gigabytes and can take many
    hours to download depending on the connection to the TCIA servers.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    # NOTE: The preprocessing below skips volumes that were converted already, so an interrupted run resumes.
    preprocessed_dir = os.path.join(path, "preprocessed")

    os.makedirs(path, exist_ok=True)

    info_path = os.path.join(path, "info.csv")
    util.download_source(path=info_path, url=URLS["info"], download=download, checksum=CHECKSUMS["info"])

    seg_dir = os.path.join(path, "segs")
    if not os.path.exists(seg_dir):
        zip_path = os.path.join(path, "segs.zip")
        util.download_source(path=zip_path, url=URLS["segs"], download=download, checksum=CHECKSUMS["segs"])
        util.unzip(zip_path=zip_path, dst=seg_dir, remove=False)

    manifest = [row for row in _read_manifest(info_path) if row["tcia_collection"] not in RESTRICTED_COLLECTIONS]
    series_uids = sorted({row["tcia_series_instance_uid"] for row in manifest})

    dicom_dir = os.path.join(path, "dicom")
    if download:  # Series that were downloaded already are skipped.
        util.download_tcia_series(series_uids, dst=dicom_dir, csv_filename=os.path.join(path, "saros"))
    elif not all(glob(os.path.join(dicom_dir, uid, "*.dcm")) for uid in series_uids):
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    # The extracted collection nests the per-case folders one level deeper (case_XXX / body-*.nii.gz).
    case_dirs = glob(os.path.join(seg_dir, "*", "case_*"))
    seg_root = os.path.dirname(case_dirs[0]) if case_dirs else seg_dir

    _preprocess_saros(seg_root, manifest, dicom_dir, preprocessed_dir)
    return preprocessed_dir


def get_saros_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the SAROS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    preprocessed_dir = get_saros_data(path, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{preprocessed_dir}'."
    return volume_paths


def get_saros_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    label_type: str = "regions",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SAROS dataset for body region or body part segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        label_type: The label volume to use, one of 'regions' (see `BODY_REGIONS`) or 'parts'
            (see `BODY_PARTS`).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert label_type in ("regions", "parts"), f"'{label_type}' is not a valid label type."
    volume_paths = get_saros_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=f"labels/{label_type}",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_saros_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    label_type: str = "regions",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SAROS dataloader for body region or body part segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        label_type: The label volume to use, one of 'regions' or 'parts'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_saros_dataset(path, patch_shape, label_type, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
