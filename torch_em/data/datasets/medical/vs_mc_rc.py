"""The Vestibular-Schwannoma-MC-RC dataset contains annotations for the vestibular schwannoma tumor in MRI.

It consists of routine clinical T1-weighted and T2-weighted MRI series of patients with a single sporadic
vestibular schwannoma, acquired at 10 sites in the United Kingdom. The tumor is annotated as a binary mask
(label id 1) in 301 series of 124 patients, each series belongs to one timepoint of a patient. Version 2 of the
collection also provides the images of 126 further timepoints, which belong to the validation and test set of a
challenge and have no public annotations, so they are not used here.

The collection has no official data splits. A split is derived from the patient ids (sorted, first 80% train,
next 10% val, last 10% test), so that all timepoints of a patient are in the same split.

The dataset is located at https://doi.org/10.7937/HRZH-2N82 and is distributed under the CC BY 4.0 license.
This dataset is from the publication https://doi.org/10.3389/fncom.2024.1365727.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = (
    "https://www.cancerimagingarchive.net/wp-content/uploads/"
    "VS-MC-RC-Segmentations_with_pathname_spreadsheet-Release2023.zip"
)
CHECKSUM = "6cbbbd10552398795bdac868385aceac70f47817bcf012d126424831ecaf198e"


def _get_labeled_series(segmentation_dir):
    import pandas as pd

    info_path = glob(os.path.join(segmentation_dir, "*", "*segmentation-info*.csv"))[0]
    label_dir = glob(os.path.join(segmentation_dir, "*", "VS-MC-RC segmentations*"))[0]
    series = pd.read_csv(info_path)
    series = series[series["SegmentationPath"].notna()]
    # The segmentations of the timepoints of the challenge validation and test set are not public.
    series = series[[os.path.exists(os.path.join(label_dir, p)) for p in series["SegmentationPath"]]]
    return series, label_dir


def _resample_labels(nifti, geometry, shape):
    """Resample the labels onto the voxel grid of the DICOM series, as their axes can be ordered differently."""
    import numpy as np
    from scipy.ndimage import map_coordinates

    labels = (np.asarray(nifti.dataobj) > 0).astype("float32")
    ras_to_voxel = np.linalg.inv(nifti.affine)
    row_step = geometry["spacing"][1] * geometry["row_direction"]
    column_step = geometry["spacing"][0] * geometry["column_direction"]

    ys, xs = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    in_plane = ys[..., None] * column_step + xs[..., None] * row_step  # (y, x, 3)
    resampled = np.zeros(shape, dtype="uint8")
    for z, origin in enumerate(geometry["origin"]):
        lps = in_plane + origin
        ras = lps * np.array([-1.0, -1.0, 1.0])
        voxel = ras @ ras_to_voxel[:3, :3].T + ras_to_voxel[:3, 3]
        resampled[z] = map_coordinates(labels, voxel.transpose(2, 0, 1), order=0, mode="constant") > 0.5
    return resampled


def _preprocess_vs_mc_rc(series, label_dir, dicom_dir, preprocessed_dir):
    import h5py
    import nibabel as nib

    os.makedirs(preprocessed_dir, exist_ok=True)
    for _, row in tqdm(series.iterrows(), total=len(series), desc="Preprocess Vestibular-Schwannoma-MC-RC"):
        patient_id, date, label_name = row["SegmentationPath"].split("/")
        modality = label_name.split("_")[-1].split(".")[0]
        out_path = os.path.join(preprocessed_dir, f"{patient_id}_{date}_{modality}.h5")
        if os.path.exists(out_path):
            continue

        volume, geometry = util.load_dicom_series(os.path.join(dicom_dir, row["series_instance_uid"]))
        labels = _resample_labels(nib.load(os.path.join(label_dir, row["SegmentationPath"])), geometry, volume.shape)

        # The file is written to a temporary path first, so an interrupted run is not mistaken for a finished one.
        tmp_path = f"{out_path}.incomplete"
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")
        os.replace(tmp_path, out_path)


def get_vs_mc_rc_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Vestibular-Schwannoma-MC-RC dataset.

    NOTE: This requires the pydicom and nibabel python packages. The DICOM series of the annotated timepoints are
    about 9 GB.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    os.makedirs(path, exist_ok=True)

    segmentation_dir = os.path.join(path, "segmentations")
    if not os.path.exists(segmentation_dir):
        zip_path = os.path.join(path, "segmentations.zip")
        util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
        util.unzip(zip_path=zip_path, dst=segmentation_dir, remove=False)

    series, label_dir = _get_labeled_series(segmentation_dir)

    dicom_dir = os.path.join(path, "dicom")
    uids = sorted(series["series_instance_uid"].unique())
    if download:
        util.download_tcia_series(uids, dst=dicom_dir, csv_filename=os.path.join(path, "vs_mc_rc_series"))
    elif not all(glob(os.path.join(dicom_dir, uid, "*.dcm")) for uid in uids):
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    preprocessed_dir = os.path.join(path, "preprocessed")
    _preprocess_vs_mc_rc(series, label_dir, dicom_dir, preprocessed_dir)
    return preprocessed_dir


def get_vs_mc_rc_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False,
) -> List[str]:
    """Get paths to the Vestibular-Schwannoma-MC-RC data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    assert split in ("train", "val", "test"), f"'{split}' is not a valid data split."

    preprocessed_dir = get_vs_mc_rc_data(path, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{preprocessed_dir}'."

    patient_ids = sorted({os.path.basename(p).split("_")[0] for p in volume_paths})
    n_train, n_val = int(0.8 * len(patient_ids)), int(0.1 * len(patient_ids))
    split_ids = {
        "train": patient_ids[:n_train],
        "val": patient_ids[n_train:n_train + n_val],
        "test": patient_ids[n_train + n_val:],
    }[split]
    return [p for p in volume_paths if os.path.basename(p).split("_")[0] in split_ids]


def get_vs_mc_rc_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Vestibular-Schwannoma-MC-RC dataset for tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_vs_mc_rc_paths(path, split, download)

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


def get_vs_mc_rc_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Vestibular-Schwannoma-MC-RC dataloader for tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_vs_mc_rc_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
