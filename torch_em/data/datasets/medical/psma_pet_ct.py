"""The PSMA-PET-CT-Lesions dataset contains annotations for tumor lesions in whole-body PSMA PET/CT.

The dataset consists of whole-body PSMA (prostate-specific membrane antigen) PET scans of prostate
cancer patients, with an expert DICOM-SEG object marking the detected tumor lesions on each PET series.
Lesions are frequently small metastatic foci, so most of a scan is background.

NOTE: This requires the pydicom python package.

The dataset is located at https://doi.org/10.7937/r7ep-3x37 and is distributed under the
CC BY 4.0 license.
Please cite it if you use this dataset in your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .adrenal_acc import _load_dicom_volume, _load_dicom_seg, _resample_labels
from .. import util


COLLECTION = "PSMA-PET-CT-Lesions"


def _get_series_metadata(path, download):
    """Get the metadata of all series in the collection from the NBIA REST API."""
    import requests

    metadata_path = os.path.join(path, "psma_pet_ct_series.json")
    if not os.path.exists(metadata_path):
        if not download:
            raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")
        response = requests.get(f"{util.NBIA_API_URL}getSeries", params={"Collection": COLLECTION})
        response.raise_for_status()
        with open(metadata_path, "w") as f:
            json.dump(response.json(), f, indent=2)

    with open(metadata_path, "r") as f:
        return json.load(f)


def _referenced_series_uid(seg):
    return str(seg.ReferencedSeriesSequence[0].SeriesInstanceUID)


def _preprocess_psma_pet_ct(dicom_dir, series_metadata, preprocessed_dir):
    import h5py
    import pydicom

    seg_series = [series for series in series_metadata if series.get("Modality") == "SEG"]

    os.makedirs(preprocessed_dir, exist_ok=True)
    for series in tqdm(seg_series, desc="Preprocess PSMA-PET-CT-Lesions"):
        seg_paths = glob(os.path.join(dicom_dir, series["SeriesInstanceUID"], "*.dcm"))
        if not seg_paths:
            continue

        out_path = os.path.join(preprocessed_dir, f"{series['SeriesInstanceUID']}.h5")
        if os.path.exists(out_path):
            continue

        seg = pydicom.dcmread(seg_paths[0])
        pet_dir = os.path.join(dicom_dir, _referenced_series_uid(seg))
        if not glob(os.path.join(pet_dir, "*.dcm")):
            continue

        volume, pet_affine = _load_dicom_volume(pet_dir)
        seg_labels, seg_affine = _load_dicom_seg(seg_paths[0])
        labels = _resample_labels(seg_labels, seg_affine, volume.shape, pet_affine)
        if labels.max() == 0:
            continue

        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")


def get_psma_pet_ct_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PSMA-PET-CT-Lesions dataset.

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

    seg_uids = [series["SeriesInstanceUID"] for series in series_metadata if series.get("Modality") == "SEG"]

    dicom_dir = os.path.join(path, "dicom")
    if download:  # The SEG series are downloaded first, so the PET series they reference can be found.
        util.download_tcia_series(seg_uids, dst=dicom_dir, csv_filename=os.path.join(path, "psma_pet_ct_seg"))

        import pydicom
        pet_uids = set()
        for uid in seg_uids:
            seg_paths = glob(os.path.join(dicom_dir, uid, "*.dcm"))
            if seg_paths:
                seg = pydicom.dcmread(seg_paths[0])
                pet_uids.add(_referenced_series_uid(seg))
        util.download_tcia_series(sorted(pet_uids), dst=dicom_dir, csv_filename=os.path.join(path, "psma_pet_ct_pet"))
    elif not glob(os.path.join(dicom_dir, "*", "*.dcm")):
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    _preprocess_psma_pet_ct(dicom_dir, series_metadata, preprocessed_dir)
    return preprocessed_dir


def get_psma_pet_ct_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get paths to the PSMA-PET-CT-Lesions data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    preprocessed_dir = get_psma_pet_ct_data(path, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{preprocessed_dir}'."
    return volume_paths


def get_psma_pet_ct_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PSMA-PET-CT-Lesions dataset for tumor lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_psma_pet_ct_paths(path, download)

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


def get_psma_pet_ct_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PSMA-PET-CT-Lesions dataloader for tumor lesion segmentation.

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
    dataset = get_psma_pet_ct_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
