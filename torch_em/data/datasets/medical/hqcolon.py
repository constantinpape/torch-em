"""HQColon is a clinically validated dataset of 435 human colons segmented from CT colonography (CTC).

The CTC volumes are from the publicly available CT Colonography collection on The Cancer Imaging
Archive (TCIA), and are downloaded here directly from TCIA by their series instance UID. For each
volume, two segmentation masks are provided: one for the entire colon (including collapsed segments
and fluid) and one for only the gas-filled parts of the colon. Both masks were generated with a
hybrid interactive machine learning pipeline and clinically validated by an expert abdominal
radiologist.

NOTE: This requires the pydicom python package.

The dataset is located at https://doi.org/10.17605/OSF.IO/8TKPM.

This dataset is from the publications https://doi.org/10.1038/s41597-025-06518-z (dataset) and
https://doi.org/10.48550/arXiv.2502.21183 (annotation pipeline). Please cite them if you use this
dataset in your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "metadata": "https://osf.io/download/8w6q7/",
    "gas_and_fluid": "https://osf.io/download/d4sc3/",
    "gas": "https://osf.io/download/y3ad2/",
}

CHECKSUMS = {
    "metadata": "158bd6b4551c07f60ba3d32c7702ef67165b03308a5e1b5fa9e943598dd77693",
    "gas_and_fluid": "99c0986b03291dbd0d4d973dc35bc5900e575fdb2f9ac9ea584381a5b12240bc",
    "gas": "04bcb14aec9c4734756853f7c4b439b7de3c4a2ee30cedf482939a633cd4d840",
}

MASK_FOLDERS = {"gas_and_fluid": "Segmentation Air and Fluid", "gas": "Segmentation Air"}


def _load_entries(metadata_path):
    with open(metadata_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _preprocess_hqcolon(path, entries, dicom_dir, preprocessed_dir):
    import SimpleITK as sitk

    os.makedirs(preprocessed_dir, exist_ok=True)
    for entry in tqdm(entries, desc="Preprocess HQColon"):
        out_path = os.path.join(preprocessed_dir, f"{entry['subject_id']}.h5")
        if os.path.exists(out_path):
            continue

        series_dir = os.path.join(dicom_dir, entry["InstanceUID"])
        if not glob(os.path.join(series_dir, "*.dcm")):
            continue

        gas_fluid_path = os.path.join(path, MASK_FOLDERS["gas_and_fluid"], entry["nnunet_label_file"])
        gas_path = os.path.join(path, MASK_FOLDERS["gas"], entry["nnunet_label_file"])
        if not (os.path.exists(gas_fluid_path) and os.path.exists(gas_path)):
            continue

        volume, _ = util.load_dicom_series(series_dir)
        labels_gas_fluid = sitk.GetArrayFromImage(sitk.ReadImage(gas_fluid_path))
        labels_gas = sitk.GetArrayFromImage(sitk.ReadImage(gas_path))

        assert volume.shape == labels_gas_fluid.shape == labels_gas.shape, \
            f"Shape mismatch for {entry['subject_id']}: {volume.shape}, {labels_gas_fluid.shape}, {labels_gas.shape}"

        import h5py
        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=volume, compression="gzip")
            f.create_dataset("labels/gas_and_fluid", data=labels_gas_fluid.astype("uint8"), compression="gzip")
            f.create_dataset("labels/gas", data=labels_gas.astype("uint8"), compression="gzip")


def get_hqcolon_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the HQColon dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    # NOTE: The preprocessing below skips volumes that were converted already, so an interrupted run resumes.
    preprocessed_dir = os.path.join(path, "preprocessed")

    os.makedirs(path, exist_ok=True)

    metadata_path = os.path.join(path, "meta-data.json")
    util.download_source(path=metadata_path, url=URLS["metadata"], download=download, checksum=CHECKSUMS["metadata"])
    entries = _load_entries(metadata_path)

    for name in ["gas_and_fluid", "gas"]:
        mask_dir = os.path.join(path, MASK_FOLDERS[name])
        if os.path.exists(mask_dir):
            continue
        zip_path = os.path.join(path, f"{name}.zip")
        util.download_source(path=zip_path, url=URLS[name], download=download, checksum=CHECKSUMS[name])
        util.unzip(zip_path=zip_path, dst=path)

    dicom_dir = os.path.join(path, "dicom")
    if download:
        series_uids = [entry["InstanceUID"] for entry in entries]
        util.download_tcia_series(series_uids, dst=dicom_dir, csv_filename=os.path.join(path, "hqcolon_series"))

    _preprocess_hqcolon(path, entries, dicom_dir, preprocessed_dir)
    return preprocessed_dir


def get_hqcolon_paths(
    path: Union[os.PathLike, str],
    label_choice: Literal["gas_and_fluid", "gas"] = "gas_and_fluid",
    download: bool = False,
) -> List[str]:
    """Get paths to the HQColon data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        label_choice: The choice of segmentation mask. Either 'gas_and_fluid' (the entire colon, including
            collapsed segments and fluid) or 'gas' (only the gas-filled parts of the colon).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data
        ('labels/gas_and_fluid' and 'labels/gas').
    """
    if label_choice not in MASK_FOLDERS:
        raise ValueError(f"'{label_choice}' is not a valid label choice. Choose from {list(MASK_FOLDERS.keys())}.")

    preprocessed_dir = get_hqcolon_data(path, download)
    volume_paths = natsorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    assert len(volume_paths) > 0, f"Could not find any preprocessed samples in '{preprocessed_dir}'."
    return volume_paths


def get_hqcolon_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    label_choice: Literal["gas_and_fluid", "gas"] = "gas_and_fluid",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HQColon dataset for colon segmentation in CT colonography.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of segmentation mask. Either 'gas_and_fluid' or 'gas'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_hqcolon_paths(path, label_choice, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=f"labels/{label_choice}",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_hqcolon_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    label_choice: Literal["gas_and_fluid", "gas"] = "gas_and_fluid",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the HQColon dataloader for colon segmentation in CT colonography.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of segmentation mask. Either 'gas_and_fluid' or 'gas'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hqcolon_dataset(path, patch_shape, label_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
