"""The M&Ms dataset contains annotations for cardiac structure segmentation in
multi-centre, multi-vendor and multi-disease cine cardiac MRI.

The data was curated for the M&Ms challenge (https://www.ub.edu/mnms), which was held at MICCAI 2020 and
targets models that generalize across clinical centres and scanner vendors. The public 'OpenDataset' release
consists of 345 of the 375 studies of the cohort: 175 training studies (150 of them annotated), 34 validation
studies and 136 testing studies, all acquired in six clinical centres with scanners of four vendors
('A' = Siemens, 'B' = Philips, 'C' = General Electric, 'D' = Canon). This module exposes the 320 annotated
studies via the official splits ('train', 'val' and 'test'), and the vendor can be selected with the
'vendor' argument.

Each study is a 4D short-axis cine acquisition, of which the end diastole (ED) and the end systole (ES) phase
are annotated. The two phases are extracted into separate volumes (selected with the 'phase' argument), so
that this module provides 640 annotated volumes, 300 of them in the training split. The label ids are
described in `LABEL_IDS`: 1 = left ventricle cavity, 2 = left ventricle myocardium, 3 = right ventricle.
NOTE: This is not the label order of the ACDC dataset, which numbers the same structures the other way round.

The cine volumes are stored as 4D nifti files with the slice axis third, so the annotated phases are
converted to hdf5 volumes with the slice axis first (the keys are 'raw' and 'labels') by this module.

NOTE: The official data at https://www.ub.edu/mnms is only handed out after signing a data use agreement, so
this module downloads a public mirror of the release at https://huggingface.co/datasets/zhuyinheng/mnms.
If the official 'OpenDataset' folder is placed in the folder passed as 'path', it is used instead.

This module covers the first edition of the challenge. M&Ms-2 (https://www.ub.edu/mnms-2), which focuses on
the right ventricle and adds long-axis views, is a different dataset and would need a separate module.

This dataset is from the publication https://doi.org/10.1109/TMI.2021.3090082.
Please cite it if you use this dataset in your research.
"""

import os
import csv
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL_BASE = "https://huggingface.co/datasets/zhuyinheng/mnms/resolve/main"

CSV_NAME = "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"

LABEL_IDS = {"background": 0, "LV": 1, "MYO": 2, "RV": 3}

SPLITS = {"train": os.path.join("Training", "Labeled"), "val": "Validation", "test": "Testing"}

PHASES = ["ED", "ES"]

VENDORS = {"A": "Siemens", "B": "Philips", "C": "General Electric", "D": "Canon"}

SUBJECT_IDS = {
    "train": [
        "A0S9V9", "A1D0Q7", "A1D9Z7", "A1E9Q1", "A1O8Z3", "A2C0I1", "A2N8V0", "A3B7E5", "A3H1O5", "A4B5U4", "A4J4S4",
        "A4U9V5", "A5E0T8", "A6B5G9", "A6D5F9", "A6M1Q7", "A7D9L8", "A7G0P5", "A7M7P8", "A7O4T6", "A8C9U8", "A8E1F4",
        "A9C5P4", "A9E3G9", "A9J5Q7", "A9J8W7", "B0I2Z0", "B0N3W8", "B2C2Z7", "B2D9M2", "B2D9O2", "B2F4K5", "B2G5R2",
        "B3D0N1", "B3O1S0", "B3P3R1", "B4O3V3", "B6D0U7", "B8H5H6", "B8J7R4", "B9E0Q1", "B9O1Q0", "C0K1P0", "C0S7W0",
        "C1G5Q0", "C1K8P5", "C2J0K3", "C2L5P7", "C2M6P8", "C3I2K3", "C4R8T7", "C4S8W9", "C5M4S2", "C6J5P1", "C8P3S7",
        "D0H9I4", "D0R0R9", "D1J5P6", "D1L4Q9", "D1M1S6", "D3D4Y5", "D3F3O5", "D3F9H9", "D3O9U9", "D4M3Q2", "D4N6W6",
        "D6H6O2", "D8E4F4", "D9L1Z3", "E0M3U7", "E0O0S0", "E3T0Z2", "E4M2Q7", "E4W8Z7", "E5E6O8", "E5F5V7", "E9H1U4",
        "E9H2K7", "E9L1W5", "E9V4Z8", "F0J2R8", "F1F3I6", "F2H5S1", "F3G5K5", "F4K3S1", "F5I9Q2", "F8N2S1", "G0H4J3",
        "G0I6P3", "G1N6S7", "G2J1M5", "G2M7W4", "G2O2S6", "G4L8Z7", "G4S9U3", "G5P4U3", "G7I5V7", "G8N2U5", "G9L0O9",
        "H0K3Q4", "H1I3W0", "H1J5W8", "H1M5Y6", "H1W2Y1", "H3U1Y1", "H4I2T8", "H5N0P0", "H6I0I6", "H7I4J3", "H7N4V9",
        "I0J5U3", "I2K2Y8", "I6N3P3", "I7T3U1", "J1T9Y1", "J4J9W6", "J6K6P5", "J6P5T8", "J8R5W2", "J9L6N9", "K2S1U6",
        "K4T7Y0", "K5L2U3", "K5P0Y1", "L1Q1Z5", "L1Q9V8", "L4Q2U3", "L5Q6T7", "M0P8U8", "M1R4S1", "M2P1R1", "M4P7Q6",
        "N1P8Q9", "N5S7Y1", "N7V9W9", "N8N9U0", "O0S9V7", "O3R8Y5", "P0S5Y0", "P5R1Y4", "P6U0Y0", "P9S7W2", "Q0U0V5",
        "Q3R9W7", "Q7V1Y5", "R4Y1Z9", "S1S3Z7", "T2T9Z9", "T9U9W2", "W5Z4Z8"
    ],
    "val": [
        "A5C2D2", "A9F3T5", "B0H7V0", "C4E9I1", "C5L0R0", "C6E0F9", "C8I7P7", "C8J7L5", "D1H6U2", "D1R0Y5", "D1S5T8",
        "D2U0V0", "D5G3W8", "D6N7Q8", "D7M8P9", "D7T3V8", "D8O0W2", "E1L8Y4", "E3F2U7", "E4O8P3", "F6J9L9", "G7Q2W0",
        "H3R6S9", "H6P7T1", "I4L4V7", "I7W4Y8", "J6M5O2", "K3R0Y7", "K5K6N1", "K5M7V5", "K9N0W0", "N2O7U5", "O7Q7U3",
        "P8V0Y7"
    ],
    "test": [
        "A1K2P5", "A2H5K9", "A2L1N6", "A3H5R1", "A3P9V7", "A4A8V9", "A4B9O6", "A4K8R4", "A4R4T0", "A5D0G0", "A5H1Q2",
        "A5P5W0", "A5Q1W8", "A6A8H0", "A6B7Y4", "A6J0Y2", "A7E4J0", "A7F4G2", "A8C5E9", "A9L7Y7", "B0L3Y2", "B1G9J3",
        "B2L0L2", "B3E2W8", "B3F0V9", "B3S2Z4", "B4E1K1", "B4S1Y2", "B5F8L9", "B5L5Y4", "B5T6V0", "B6I0T4", "B7F5P0",
        "B8P5Q9", "B9G4U2", "B9H8N8", "C0L7V1", "C0N8P4", "C6U4W8", "C7L8Z8", "C7M6W0", "C8O0P2", "D1H2O9", "D1L6T4",
        "D3K5Q2", "D3Q0W9", "D6E9U8", "D9F5P1", "D9I8O7", "E0J2Z9", "E0J7L9", "E1L7M3", "E3F5U2", "E3L8U8", "E4H7L4",
        "E4I9O7", "E5J6L2", "E5S7W7", "E6H0V9", "E6J4N8", "E6M6P2", "E7L0N6", "E9V9Z2", "F0I6U8", "F0K4T6", "F1K2S9",
        "F5I1Z8", "F9M1R2", "G1J5K3", "G1K1V3", "G3M5S4", "G4I7V2", "G7N8R7", "G7S6V0", "G8K0M3", "G8L0Z0", "G8R0Z9",
        "G9N5V9", "H1N8S6", "H2M9S1", "H7K5U5", "H7L8R8", "H7P5Z4", "H8K2K7", "I0I2J8", "I2J6Z6", "I4R8V6", "I5L3S2",
        "I6P4R0", "I6T4W8", "I8N8Y1", "I8Z0Z6", "J4J8Q3", "J6K4V3", "J9L4S2", "K3P3Y6", "K5L4S1", "K6N4N7", "K7L2Y6",
        "K7N0R7", "K7O3Q0", "L2V5Z0", "L5U7Y4", "L6T2T5", "L7Y7Z2", "L8M2U8", "L8N7P0", "L8N7Z0", "M2P5T8", "M4T4V6",
        "M6M9N1", "M6V2Y0", "N7P3T8", "N7W6Z8", "N9P5Z0", "N9Q4T8", "O4O6U5", "O4T6Y7", "O5U2U7", "O9V8W5", "P3P9S5",
        "P3R6Y5", "P3T5U1", "P8W4Z0", "Q1Q3T1", "Q3Q6R8", "Q4W5Z8", "Q5V8W3", "R1R6Y8", "R2R7Z5", "R3V5W7", "R6V5W3",
        "R8V0Y4", "T2Z1Z9", "V4W8Z5", "Y6Y9Z2"
    ],
}


def _get_metadata(path, download):
    """Read the official information sheet, which holds the vendor and the ED and ES frame index per subject."""
    csv_paths = glob(os.path.join(path, "**", CSV_NAME), recursive=True)
    csv_path = csv_paths[0] if csv_paths else os.path.join(path, CSV_NAME)
    util.download_source(path=csv_path, url=f"{URL_BASE}/{CSV_NAME.replace('&', '%26')}", download=download)

    with open(csv_path, "r") as f:
        return {row["External code"]: row for row in csv.DictReader(f)}


def _download_volumes(path, split, subject_ids):
    split_dir = os.path.join(path, "OpenDataset", SPLITS[split])
    for subject_id in tqdm(subject_ids, desc=f"Downloading the M&Ms studies of '{split}'"):
        for suffix in ["sa", "sa_gt"]:
            volume_path = os.path.join(split_dir, subject_id, f"{subject_id}_{suffix}.nii.gz")
            os.makedirs(os.path.dirname(volume_path), exist_ok=True)
            url = f"{URL_BASE}/{SPLITS[split].replace(os.sep, '/')}/{subject_id}/{subject_id}_{suffix}.nii.gz"
            util.download_source(path=volume_path, url=url, download=True)

    return split_dir


def _preprocess_inputs(split_dir, metadata, subject_ids, preprocessed_dir):
    import h5py
    import nibabel as nib

    os.makedirs(preprocessed_dir, exist_ok=True)
    for subject_id in tqdm(subject_ids, desc="Preprocessing the M&Ms studies"):
        raw_volume = nib.load(os.path.join(split_dir, subject_id, f"{subject_id}_sa.nii.gz"))
        label_volume = nib.load(os.path.join(split_dir, subject_id, f"{subject_id}_sa_gt.nii.gz"))

        for phase in PHASES:
            volume_path = os.path.join(preprocessed_dir, f"{subject_id}_{phase}.h5")
            if os.path.exists(volume_path):
                continue

            # The transpose maps the nifti axis order (X, Y, Z) of one phase to the (Z, Y, X) order of the volumes.
            frame = int(metadata[subject_id][phase])
            raw = np.asarray(raw_volume.dataobj[..., frame]).T
            labels = np.asarray(label_volume.dataobj[..., frame]).T

            # The file is written to a temporary path first, so that an interrupted run leaves no corrupt file.
            with h5py.File(f"{volume_path}.tmp", "w") as f:
                f.create_dataset("raw", data=raw, compression="gzip")
                f.create_dataset("labels", data=labels.astype("uint8"), compression="gzip")

            os.rename(f"{volume_path}.tmp", volume_path)


def get_mnms_data(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False
) -> str:
    """Download the M&Ms dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Please choose one of {list(SPLITS.keys())}.")

    subject_ids = SUBJECT_IDS[split]
    preprocessed_dir = os.path.join(path, "preprocessed", split)
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == len(subject_ids) * len(PHASES):
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    # Use the official release if it was downloaded manually, otherwise fetch the studies from the mirror.
    mirror_dir = os.path.join(path, "OpenDataset", SPLITS[split])
    split_dirs = [p for p in glob(os.path.join(path, "**", SPLITS[split]), recursive=True) if p != mirror_dir]
    if split_dirs:
        split_dir = split_dirs[0]
    elif download:
        split_dir = _download_volumes(path, split, subject_ids)
    else:
        raise RuntimeError(f"Cannot find the data at '{path}', but download was set to False.")

    _preprocess_inputs(split_dir, _get_metadata(path, download), subject_ids, preprocessed_dir)
    return preprocessed_dir


def get_mnms_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    phase: Optional[Literal["ED", "ES"]] = None,
    vendor: Optional[Literal["A", "B", "C", "D"]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the M&Ms data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        phase: The choice of cardiac phase. Either 'ED' or 'ES'. If None, both phases are returned.
        vendor: The choice of scanner vendor. One of 'A', 'B', 'C' or 'D'. If None, all vendors are returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_mnms_data(path, split, download)

    if phase is not None and phase not in PHASES:
        raise ValueError(f"'{phase}' is not a valid phase. Please choose one of {PHASES}.")

    volume_paths = natsorted(glob(os.path.join(data_dir, f"*_{'*' if phase is None else phase}.h5")))

    if vendor is not None:
        if vendor not in VENDORS:
            raise ValueError(f"'{vendor}' is not a valid vendor. Please choose one of {list(VENDORS.keys())}.")

        metadata = _get_metadata(path, download)
        volume_paths = [
            p for p in volume_paths if metadata[os.path.basename(p).rsplit("_", 1)[0]]["Vendor"] == vendor
        ]

    assert len(volume_paths) > 0, f"Could not find any preprocessed volumes in '{data_dir}'."
    return volume_paths


def get_mnms_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    phase: Optional[Literal["ED", "ES"]] = None,
    vendor: Optional[Literal["A", "B", "C", "D"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the M&Ms dataset for cardiac structure segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        phase: The choice of cardiac phase. Either 'ED' or 'ES'. If None, both phases are returned.
        vendor: The choice of scanner vendor. One of 'A', 'B', 'C' or 'D'. If None, all vendors are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_mnms_paths(path, split, phase, vendor, download)

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


def get_mnms_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    phase: Optional[Literal["ED", "ES"]] = None,
    vendor: Optional[Literal["A", "B", "C", "D"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the M&Ms dataloader for cardiac structure segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        phase: The choice of cardiac phase. Either 'ED' or 'ES'. If None, both phases are returned.
        vendor: The choice of scanner vendor. One of 'A', 'B', 'C' or 'D'. If None, all vendors are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mnms_dataset(path, patch_shape, split, phase, vendor, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
