"""The Multi-organ Abdominal CT dataset contains reference standard annotations for multiple abdominal organs
in CT, released with the paper 'Automatic Multi-organ Segmentation on Abdominal CT with Dense V-networks'.

The dataset comprises annotations for 90 abdominal CT volumes: 43 volumes from the TCIA Pancreas-CT collection
(one of them, PANCREAS_0025, has since been removed from TCIA, so 42 are available) and 47 volumes from the
Beyond the Cranial Vault (BTCV) abdomen challenge. The Zenodo record contains only the annotations,
the CT volumes have to be obtained from the original sources:
- The TCIA Pancreas-CT volumes are public and downloaded automatically from the TCIA NBIA API.
- The BTCV volumes require registration at Synapse. Please download 'RawData.zip' from
  https://www.synapse.org/#!Synapse:syn3193805 (Abdomen) and place it in the folder passed as 'path'.

NOTE: The label legend is as follows (labels marked with * are only annotated in the BTCV volumes):
- background: 0, spleen: 1, right kidney*: 2, left kidney: 3, gallbladder: 4, esophagus: 5, liver: 6, stomach: 7,
  aorta*: 8, inferior vena cava*: 9, portal vein and splenic vein*: 10, pancreas: 11, right adrenal gland*: 12,
  left adrenal gland*: 13, duodenum: 14
The annotations may be incomplete outside of the cropping region specified in 'cropping.csv' (see the Zenodo record).

The dataset is located at https://zenodo.org/records/1169361.

This dataset is from the publication https://doi.org/10.1109/TMI.2018.2806309.
Please cite it if you use this dataset in your research.
"""

import os
import shutil
from glob import glob
from warnings import warn
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np
import requests

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "tcia": "https://zenodo.org/records/1169361/files/label_tciapancreasct_multiorgan.tar.gz?download=1",
    "btcv": "https://zenodo.org/records/1169361/files/label_btcv_multiorgan.tar.gz?download=1",
    "cropping": "https://zenodo.org/records/1169361/files/cropping.csv?download=1",
}

CHECKSUMS = {
    "tcia": "1790e252ba0732cc06ec727f00245c8d9c1d5f0bb9829fd3e9915863c5100c61",
    "btcv": "bb080c7de1094cc0ee46a5e1bef5b66e635075f8e9a3080ca681086cd2723998",
    "cropping": "d6503bda3d776c10698523aae14446409bc9d3722fb405b0025b4a5f59094f8f",
}

NBIA_API_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
TCIA_COLLECTION = "Pancreas-CT"

# NOTE: The case PANCREAS_0025 has been removed from the TCIA Pancreas-CT collection, so its annotation is skipped.
MISSING_TCIA_CASES = ["0025"]

LABEL_DIRS = {"tcia": "label_tcia_multiorgan", "btcv": "label_btcv_multiorgan"}
IMAGE_DIRS = {"tcia": "image_tcia_multiorgan", "btcv": "image_btcv_multiorgan"}

ORGANS = {
    "spleen": 1, "right kidney": 2, "left kidney": 3, "gallbladder": 4, "esophagus": 5, "liver": 6, "stomach": 7,
    "aorta": 8, "inferior vena cava": 9, "portal vein and splenic vein": 10, "pancreas": 11,
    "right adrenal gland": 12, "left adrenal gland": 13, "duodenum": 14,
}


def _get_label_id(label_path):
    return os.path.basename(label_path).replace("label", "").replace(".nii.gz", "")


def _download_labels(path, source, download):
    label_dir = os.path.join(path, LABEL_DIRS[source])
    if not os.path.exists(label_dir):
        tar_path = os.path.join(path, f"{LABEL_DIRS[source]}.tar.gz")
        util.download_source(path=tar_path, url=URLS[source], download=download, checksum=CHECKSUMS[source])
        util.unzip_tarfile(tar_path=tar_path, dst=path)

    csv_path = os.path.join(path, "cropping.csv")
    if not os.path.exists(csv_path):
        util.download_source(path=csv_path, url=URLS["cropping"], download=download, checksum=CHECKSUMS["cropping"])

    return label_dir


def _get_tcia_series_uids():
    response = requests.get(f"{NBIA_API_URL}/getSeries", params={"Collection": TCIA_COLLECTION})
    response.raise_for_status()
    return {series["PatientID"]: series["SeriesInstanceUID"] for series in response.json()}


def _convert_tcia_dicom_to_nifti(dicom_dir, label_path, image_path):
    import imageio.v2 as imageio
    import nibabel as nib

    # The DICOM slices are sorted by their position and stacked to a (Z, Y, X) volume,
    # which is transposed to match the (X, Y, Z) axis order of the nifti annotations.
    volume = np.asarray(imageio.volread(dicom_dir, format="DICOM"))
    volume = volume.transpose(2, 1, 0)

    # The reference annotations share the geometry of the CT volumes, so we store the CT with the label's affine.
    label = nib.load(label_path)
    assert volume.shape == label.shape, f"Shape mismatch between CT {volume.shape} and label {label.shape}."
    nib.save(nib.Nifti1Image(volume, label.affine), image_path)


def _get_label_paths(label_dir, source):
    label_paths = natsorted(glob(os.path.join(label_dir, "label*.nii.gz")))
    if source == "tcia":
        label_paths = [p for p in label_paths if _get_label_id(p) not in MISSING_TCIA_CASES]
    return label_paths


def _prepare_tcia_images(path, label_dir, download):
    image_dir = os.path.join(path, IMAGE_DIRS["tcia"])
    label_paths = _get_label_paths(label_dir, "tcia")
    if len(glob(os.path.join(image_dir, "*.nii.gz"))) == len(label_paths):
        return image_dir

    if not download:
        raise RuntimeError(f"Cannot find the CT volumes at {image_dir}, but download was set to False.")

    os.makedirs(image_dir, exist_ok=True)
    dicom_root = os.path.join(path, "tcia_dicom")
    os.makedirs(dicom_root, exist_ok=True)

    series_uids = _get_tcia_series_uids()
    for label_path in tqdm(label_paths, desc="Download and convert the TCIA Pancreas-CT volumes"):
        label_id = _get_label_id(label_path)
        image_path = os.path.join(image_dir, f"img{label_id}.nii.gz")
        if os.path.exists(image_path):
            continue

        patient_id = f"PANCREAS_{label_id}"
        if patient_id not in series_uids:
            warn(f"The case '{patient_id}' is not available in the TCIA Pancreas-CT collection and will be skipped.")
            continue

        zip_path = os.path.join(dicom_root, f"{patient_id}.zip")
        dicom_dir = os.path.join(dicom_root, patient_id)
        if not os.path.exists(dicom_dir):
            url = f"{NBIA_API_URL}/getImage?SeriesInstanceUID={series_uids[patient_id]}"
            util.download_source(path=zip_path, url=url, download=download, checksum=None)
            util.unzip(zip_path=zip_path, dst=dicom_dir)

        _convert_tcia_dicom_to_nifti(dicom_dir, label_path, image_path)
        # The DICOM files are removed after the conversion to save disk space.
        shutil.rmtree(dicom_dir)

    return image_dir


def _prepare_btcv_images(path, label_dir):
    image_dir = os.path.join(path, IMAGE_DIRS["btcv"])
    label_paths = _get_label_paths(label_dir, "btcv")
    if len(glob(os.path.join(image_dir, "*.nii.gz"))) == len(label_paths):
        return image_dir

    zip_path = os.path.join(path, "RawData.zip")
    if not os.path.exists(zip_path):
        raise RuntimeError(
            "The BTCV CT volumes cannot be downloaded automatically. Please register at Synapse, join the challenge "
            "at https://www.synapse.org/#!Synapse:syn3193805 and download 'RawData.zip' from the 'Abdomen' folder "
            f"at https://www.synapse.org/#!Synapse:syn3376386. Place the file at '{zip_path}' and try again."
        )

    raw_dir = os.path.join(path, "btcv_raw")
    if not os.path.exists(raw_dir):
        util.unzip(zip_path=zip_path, dst=raw_dir, remove=False)

    os.makedirs(image_dir, exist_ok=True)
    for label_path in label_paths:
        label_id = _get_label_id(label_path)
        image_path = os.path.join(image_dir, f"img{label_id}.nii.gz")
        if os.path.exists(image_path):
            continue

        # The BTCV ids 0001-0040 are part of the challenge training set, 0061-0080 of the challenge test set.
        candidates = glob(os.path.join(raw_dir, "**", f"img{label_id}.nii.gz"), recursive=True)
        if len(candidates) != 1:
            raise RuntimeError(f"Could not find the BTCV volume 'img{label_id}.nii.gz' in '{raw_dir}'.")
        os.symlink(os.path.abspath(candidates[0]), image_path)

    return image_dir


def get_multi_organ_abdominal_ct_data(
    path: Union[os.PathLike, str], source: Literal["tcia", "btcv"], download: bool = False
) -> Tuple[str, str]:
    """Download the Multi-organ Abdominal CT dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        source: The source of the CT volumes. Either 'tcia' (Pancreas-CT) or 'btcv' (Beyond the Cranial Vault).
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the CT volumes are stored.
        Filepath where the annotations are stored.
    """
    if source not in LABEL_DIRS:
        raise ValueError(f"'{source}' is not a valid source. Please choose from {list(LABEL_DIRS.keys())}.")

    os.makedirs(path, exist_ok=True)
    label_dir = _download_labels(path, source, download)

    if source == "tcia":
        image_dir = _prepare_tcia_images(path, label_dir, download)
    else:
        image_dir = _prepare_btcv_images(path, label_dir)

    return image_dir, label_dir


def get_multi_organ_abdominal_ct_paths(
    path: Union[os.PathLike, str], source: Literal["tcia", "btcv"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Multi-organ Abdominal CT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        source: The source of the CT volumes. Either 'tcia' (Pancreas-CT) or 'btcv' (Beyond the Cranial Vault).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    image_dir, label_dir = get_multi_organ_abdominal_ct_data(path, source, download)

    label_paths = _get_label_paths(label_dir, source)
    raw_paths = [os.path.join(image_dir, f"img{_get_label_id(p)}.nii.gz") for p in label_paths]
    assert all(os.path.exists(p) for p in raw_paths)

    return raw_paths, label_paths


def get_multi_organ_abdominal_ct_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    source: Literal["tcia", "btcv"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Multi-organ Abdominal CT dataset for abdominal organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        source: The source of the CT volumes. Either 'tcia' (Pancreas-CT) or 'btcv' (Beyond the Cranial Vault).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_multi_organ_abdominal_ct_paths(path, source, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_multi_organ_abdominal_ct_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    source: Literal["tcia", "btcv"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Multi-organ Abdominal CT dataloader for abdominal organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        source: The source of the CT volumes. Either 'tcia' (Pancreas-CT) or 'btcv' (Beyond the Cranial Vault).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_multi_organ_abdominal_ct_dataset(path, patch_shape, source, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
