"""The CTSpinoPelvic1K dataset contains annotations for the spine, pelvis, ribs and femora in CT scans.

The dataset consists of 802 CT COLONOGRAPHY scans from TCIA with 69-class annotations (see
`CLASS_NAMES`) in one coordinate frame: the cervical to lumbar vertebrae (with a sixth lumbar
vertebra where present), the sacrum, coccyx, hips, femora, the individual ribs (per side and per
level, including a rib on a lumbar vertebra where present) and spinal hardware. The vertebral
annotations derive from CTSpine1K and the pelvic annotations from CTPelvic1K, remapped onto one
coordinate frame and paired with the exact CT series they were drawn on, which neither of those
releases published. The manifest also carries a lumbosacral transitional anatomy label and a
Castellvi grade per case.

NOTE: The images are not distributed with the release: they are 193 GB against 1.8 GB of labels
and already public on TCIA, so this module downloads and reconstructs them, following the same
steps and settings as the release. A raw DICOM conversion (e.g. with dcm2niix) does not share the
label's grid, so it is resampled onto it: DICOM patient coordinates are LPS, the label is stored
in a RAS+ world frame (as dcm2niix would produce), and a trilinear resampling with a -1024 HU fill
value outside the original extent completes the match.

NOTE: This requires the pydicom and scipy python packages.

The dataset is located at https://doi.org/10.5281/zenodo.22642578 and is distributed under the
CC BY-NC-SA 4.0 license for research use only.
This dataset is from the publications https://doi.org/10.48550/arXiv.2105.14711 (CTSpine1K) and
https://doi.org/10.1007/s11548-021-02363-8 (CTPelvic1K); please cite the Zenodo record and both
publications if you use this dataset in your research.
"""

import os
import json
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
    "manifest": "https://zenodo.org/records/22642578/files/manifest.json?download=1",
    "labels": "https://zenodo.org/records/22642578/files/labels.zip?download=1",
}

CHECKSUMS = {
    "manifest": "6f25aac0ea6f4b46801d372f3d4762f05cd0dceb05da04ac73a3ad54dd427c55",
    "labels": "a6d0df210fea4660095dc27d9caf28120ceb35313daeb2b5762587fd458383f6",
}

IGNORE_LABEL = 255
"""The sentinel that marks a voxel excluded from the annotation, e.g. outside a partial scan."""

CLASS_NAMES = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
    "L1", "L2", "L3", "L4", "L5", "L6",
    "sacrum", "coccyx", "T13", "S1", "left_hip", "right_hip", "femur_left", "femur_right",
] + [f"rib_left_{i}" for i in range(1, 14)] + [f"rib_right_{i}" for i in range(1, 14)] + [
    "rib_left_lumbar", "rib_right_lumbar", "hardware", "hardware_cage", "hardware_screw_rod",
    "hardware_plate", "hardware_arthroplasty", "hardware_si_screw", "hardware_osteosynthesis",
]
"""The 68 foreground classes of the CTSpinoPelvic1K dataset. The label id of a class is its
1-based index; 255 marks an excluded voxel rather than a class. See also `CLASS_IDS`."""

CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
"""Mapping from the anatomical structure to its label id."""


def _series_uid(record):
    """The UID the labels of a record were drawn on: the spine series, or the pelvic one for the
    pelvis-only records that have no spine annotation.

    A handful of records suffix the UID with '_orientation_fixed', a note from the release's own
    pipeline rather than part of the UID itself, which is stripped to get a downloadable series UID.
    """
    uid = str(record.get("spine_series_uid") or "").strip() or str(record.get("pelvic_series_uid") or "").strip()
    return uid.removesuffix("_orientation_fixed")


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


def _preprocess_ctspinopelvic1k(label_dir, manifest_path, dicom_dir, preprocessed_dir):
    import h5py
    import nibabel as nib

    with open(manifest_path) as f:
        manifest = json.load(f)
    records = manifest if isinstance(manifest, list) else manifest.get("records", list(manifest.values()))

    os.makedirs(preprocessed_dir, exist_ok=True)
    for record in tqdm(records, desc="Preprocess CTSpinoPelvic1K"):
        case_id = os.path.basename(record["label_file"]).split("_")[0]
        out_path = os.path.join(preprocessed_dir, f"{case_id}.h5")
        if os.path.exists(out_path):
            continue

        label_path = os.path.join(label_dir, f"{case_id}_label.nii.gz")
        if not os.path.exists(label_path):
            continue

        series_dir = os.path.join(dicom_dir, _series_uid(record))
        if not glob(os.path.join(series_dir, "*.dcm")):
            continue

        label_image = nib.load(label_path)
        labels = np.asarray(label_image.dataobj)
        volume, ct_affine = _load_dicom_volume(series_dir)
        raw = _resample_to_label(volume, ct_affine, labels.shape, label_image.affine)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels.astype("uint8"), compression="gzip")


def get_ctspinopelvic1k_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CTSpinoPelvic1K dataset.

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

    manifest_path = os.path.join(path, "manifest.json")
    util.download_source(
        path=manifest_path, url=URLS["manifest"], download=download, checksum=CHECKSUMS["manifest"]
    )

    label_dir = os.path.join(path, "labels")
    if not os.path.exists(label_dir):
        zip_path = os.path.join(path, "labels.zip")
        util.download_source(path=zip_path, url=URLS["labels"], download=download, checksum=CHECKSUMS["labels"])
        util.unzip(zip_path=zip_path, dst=path, remove=False)

    with open(manifest_path) as f:
        manifest = json.load(f)
    records = manifest if isinstance(manifest, list) else manifest.get("records", list(manifest.values()))
    series_uids = sorted({_series_uid(record) for record in records if _series_uid(record)})

    dicom_dir = os.path.join(path, "dicom")
    if download:  # Series that were downloaded already are skipped.
        util.download_tcia_series(series_uids, dst=dicom_dir, csv_filename=os.path.join(path, "ctspinopelvic1k"))
    elif not all(glob(os.path.join(dicom_dir, uid, "*.dcm")) for uid in series_uids):
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    _preprocess_ctspinopelvic1k(label_dir, manifest_path, dicom_dir, preprocessed_dir)
    return preprocessed_dir


def get_ctspinopelvic1k_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the CTSpinoPelvic1K data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    preprocessed_dir = get_ctspinopelvic1k_data(path, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{preprocessed_dir}'."
    return volume_paths


def get_ctspinopelvic1k_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CTSpinoPelvic1K dataset for spine, pelvis, rib and femur segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_ctspinopelvic1k_paths(path, download)

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


def get_ctspinopelvic1k_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CTSpinoPelvic1K dataloader for spine, pelvis, rib and femur segmentation.

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
    dataset = get_ctspinopelvic1k_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
