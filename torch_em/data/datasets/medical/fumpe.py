"""The FUMPE dataset contains annotations for pulmonary embolism in computed tomography angiography.

The dataset consists of 35 CT angiography scans of different patients, with a binary mask marking the
pulmonary emboli. The scans are distributed as DICOM series and the masks as MATLAB files, which this
module converts into hdf5 files, one per patient.

NOTE: This requires the pydicom and scipy python packages.

NOTE: The slices of a scan are ordered by their instance number, which is the order the masks were
created in. Two of the scans are acquired from head to feet, so sorting the slices along the slice
normal instead would flip their masks.

The dataset is located at https://www.kaggle.com/datasets/andrewmvd/pulmonary-embolism-in-ct-images.
This dataset is from the publication https://doi.org/10.1038/sdata.2018.180.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


KAGGLE_DATASET_NAME = "andrewmvd/pulmonary-embolism-in-ct-images"


def _load_dicom_volume(series_dir):
    """Stack a DICOM series into a volume with axes (z, y, x), ordered by the instance number.

    The masks of this dataset were created in this order, so the slices must not be sorted along the
    slice normal, which would flip the scans that are acquired from head to feet.
    """
    import pydicom

    slices = [pydicom.dcmread(p) for p in natsorted(glob(os.path.join(series_dir, "*.dcm")))]
    slices.sort(key=lambda dcm: int(dcm.InstanceNumber))

    volume = np.stack([dcm.pixel_array for dcm in slices]).astype("float32")
    volume = volume * float(slices[0].RescaleSlope) + float(slices[0].RescaleIntercept)
    return np.round(volume).astype("int16")


def _preprocess_fumpe(data_dir, preprocessed_dir):
    import h5py
    import scipy.io as sio

    os.makedirs(preprocessed_dir, exist_ok=True)
    patient_dirs = natsorted(glob(os.path.join(data_dir, "CT_scans", "*")))
    for patient_dir in tqdm(patient_dirs, desc="Preprocess FUMPE"):
        patient_id = os.path.basename(patient_dir)
        out_path = os.path.join(preprocessed_dir, f"{patient_id}.h5")
        if os.path.exists(out_path):
            continue

        volume = _load_dicom_volume(patient_dir)
        # The masks are stored with axes (y, x, z) and have to be transposed to match the volume.
        labels = sio.loadmat(os.path.join(data_dir, "GroundTruth", f"{patient_id}.mat"))["Mask"]
        labels = np.transpose(labels, (2, 0, 1)).astype("uint8")
        assert labels.shape == volume.shape, f"The mask of '{patient_id}' does not match its scan."

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_fumpe_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the FUMPE dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if os.path.exists(preprocessed_dir):
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)
    data_dir = os.path.join(path, "FUMPE")
    if not os.path.exists(data_dir):
        util.download_source_kaggle(path=path, dataset_name=KAGGLE_DATASET_NAME, download=download)
        util.unzip(zip_path=os.path.join(path, "pulmonary-embolism-in-ct-images.zip"), dst=path, remove=False)

    _preprocess_fumpe(data_dir, preprocessed_dir)
    return preprocessed_dir


def get_fumpe_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the FUMPE data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    preprocessed_dir = get_fumpe_data(path, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{preprocessed_dir}'."
    return volume_paths


def get_fumpe_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the FUMPE dataset for pulmonary embolism segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_fumpe_paths(path, download)

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


def get_fumpe_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the FUMPE dataloader for pulmonary embolism segmentation.

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
    dataset = get_fumpe_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
