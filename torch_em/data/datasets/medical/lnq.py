"""The LNQ dataset contains annotations for mediastinal lymph node segmentation in thorax CT scans.

The dataset was curated for the LNQ2023 MICCAI challenge (https://lnq2023.grand-challenge.org).
It comprises 513 CT scans of cancer patients, each with a DICOM SEG object annotating the clinically
relevant mediastinal lymph nodes (larger than 1 cm). The annotations are partial for most cases,
i.e. not every lymph node in a scan is annotated. The data is hosted on TCIA as the
'Mediastinal-Lymph-Node-SEG' collection (https://doi.org/10.7937/QVAZ-JA09) under the CC BY 4.0 license.

The DICOM series are converted to hdf5 files with the keys 'raw' (the CT in HU) and 'labels'.
The label ids are: 0 = background, 1 = lymph node. Note that each scan is annotated with a single
segment that covers all annotated lymph nodes, i.e. the individual nodes are not separated.

The dataset is from the publication https://doi.org/10.59275/j.melba.2025-1gb5.
Please cite it if you use this dataset in your research.

NOTE: The DICOM conversion requires 'pydicom'. Install it with 'pip install pydicom'.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://www.cancerimagingarchive.net/wp-content/uploads/Mediastinal-Lymph-Node-SEG-DA-RAD.tcia"

LABEL_IDS = {"background": 0, "lymph_node": 1}


def _load_ct_series(series_dir):
    import pydicom

    slices = [pydicom.dcmread(p) for p in glob(os.path.join(series_dir, "*.dcm"))]
    slices = [s for s in slices if hasattr(s, "ImagePositionPatient")]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    z_positions = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])  # (Z, Y, X)

    slope = float(getattr(slices[0], "RescaleSlope", 1.0))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    volume = (volume * slope + intercept).astype(np.int16)

    return volume, z_positions


def _load_seg_series(series_dir, shape, z_positions):
    import pydicom

    seg_path = glob(os.path.join(series_dir, "*.dcm"))[0]
    seg = pydicom.dcmread(seg_path)

    frames = seg.pixel_array
    if frames.ndim == 2:  # A single frame is returned without the frame axis.
        frames = frames[None]

    labels = np.zeros(shape, dtype=np.uint8)
    for frame, frame_info in zip(frames, seg.PerFrameFunctionalGroupsSequence):
        frame_z = float(frame_info.PlanePositionSequence[0].ImagePositionPatient[2])

        # Match the frame to the CT slice with the closest z-position.
        z = int(np.argmin(np.abs(z_positions - frame_z)))
        assert abs(z_positions[z] - frame_z) < 1.0, "Could not match the segmentation frame to a CT slice."
        assert frame.shape == shape[1:], f"Frame shape {frame.shape} does not match the CT shape {shape[1:]}."

        labels[z][frame > 0] = 1

    return labels


def _preprocess_inputs(path, dicom_dir, csv_path):
    import h5py

    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    for case_id, case_df in tqdm(df.groupby("Subject ID"), desc="Convert the LNQ DICOM series to hdf5"):
        volume_path = os.path.join(preprocessed_dir, f"{case_id}.h5")
        if os.path.exists(volume_path):
            continue

        ct_uid = case_df[case_df["Modality"] == "CT"]["Series UID"].iloc[0]
        seg_uid = case_df[case_df["Modality"] == "SEG"]["Series UID"].iloc[0]

        raw, z_positions = _load_ct_series(os.path.join(dicom_dir, ct_uid))
        labels = _load_seg_series(os.path.join(dicom_dir, seg_uid), raw.shape, z_positions)

        # The file is written to a temporary path first, so that an interrupted run does not leave a corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)

    return preprocessed_dir


def get_lnq_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the LNQ dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded and preprocessed.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if os.path.exists(preprocessed_dir) and len(glob(os.path.join(preprocessed_dir, "*.h5"))) == 513:
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    dicom_dir = os.path.join(path, "dicom")
    csv_path = os.path.join(path, "metadata")
    if not os.path.exists(f"{csv_path}.csv"):
        util.download_source_tcia(
            path=os.path.join(path, "Mediastinal-Lymph-Node-SEG-DA-RAD.tcia"),
            url=URL,
            dst=dicom_dir,
            csv_filename=csv_path,
            download=download,
        )

    return _preprocess_inputs(path, dicom_dir, f"{csv_path}.csv")


def get_lnq_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the LNQ data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_lnq_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths, volume_paths


def get_lnq_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LNQ dataset for mediastinal lymph node segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_lnq_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="raw",
        label_paths=label_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_lnq_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LNQ dataloader for mediastinal lymph node segmentation.

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
    dataset = get_lnq_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
