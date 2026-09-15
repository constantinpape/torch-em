"""The PDDCA dataset contains annotations for organs-at-risk segmentation in head and neck CT.

The dataset (Public Domain Database for Computational Anatomy, version 1.4.1) consists of 48 CT scans from the
RTOG 0522 clinical trial with manual segmentations of nine organs at risk. It was used for the
MICCAI 2015 Head and Neck Auto-Segmentation Challenge.

NOTE: The per-structure binary masks are combined into one semantic label volume with the following ids:
- background: 0
- brainstem: 1
- optic chiasm: 2
- mandible: 3
- left optic nerve: 4
- right optic nerve: 5
- left parotid gland: 6
- right parotid gland: 7
- left submandibular gland: 8
- right submandibular gland: 9
A few structures overlap by a handful of voxels. In this case, the smaller structure takes priority:
the masks are written in the order mandible, parotids, submandibular glands, brainstem, optic nerves, chiasm,
so that a later structure overwrites an earlier one. Not all structures are annotated for every patient
(e.g. some patients lack the mandible or the right submandibular gland); missing structures are left as background.

The official challenge sub-packages are exposed as splits: 'train' (25 scans, 0522c0001 to 0522c0328),
'train_additional' (8 scans, 0522c0329 to 0522c0479), 'test_offsite' (10 scans, 0522c0555 to 0522c0746)
and 'test_onsite' (5 scans, 0522c0788 to 0522c0878).

The dataset is located at https://www.imagenglab.com/newsite/pddca/.

This dataset is from the publication https://doi.org/10.1002/mp.12197.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List, Optional

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = [
    "https://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part1.zip",
    "https://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part2.zip",
    "https://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part3.zip",
]

CHECKSUMS = [
    "5b47b94f1e6aaad0a10c694906bd40ee8bcfc701e6d3bcbe3f12005815b721ef",
    "a5354e5373050f958e2696088a24f25f72361844d8b0d6988c16a3c989fd5659",
    "58bc70db301d3f0fd2e856a33e7d2be929859bc5b1a37d7c83a47f2bbe6a7fb6",
]

LABEL_IDS = {
    "BrainStem": 1,
    "Chiasm": 2,
    "Mandible": 3,
    "OpticNerve_L": 4,
    "OpticNerve_R": 5,
    "Parotid_L": 6,
    "Parotid_R": 7,
    "Submandibular_L": 8,
    "Submandibular_R": 9,
}

# The order in which the structures are written to the label volume. Later structures overwrite earlier ones.
WRITE_ORDER = [
    "Mandible", "Parotid_L", "Parotid_R", "Submandibular_L", "Submandibular_R",
    "BrainStem", "OpticNerve_L", "OpticNerve_R", "Chiasm",
]

SPLITS = {
    "train": (1, 328),
    "train_additional": (329, 479),
    "test_offsite": (555, 746),
    "test_onsite": (788, 878),
}


def _convert_case(case_dir, out_path):
    import nrrd
    import h5py

    raw, header = nrrd.read(os.path.join(case_dir, "img.nrrd"))
    labels = np.zeros(raw.shape, dtype="uint8")
    for name in WRITE_ORDER:
        mask_path = os.path.join(case_dir, "structures", f"{name}.nrrd")
        if not os.path.exists(mask_path):
            continue
        mask, _ = nrrd.read(mask_path)
        assert mask.shape == raw.shape, f"Shape mismatch for {mask_path}."
        labels[mask > 0] = LABEL_IDS[name]

    # The nrrd arrays are stored in (x, y, z) order, we transpose them to (z, y, x).
    raw, labels = raw.transpose(2, 1, 0), labels.transpose(2, 1, 0)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
        f.attrs["spacing"] = np.diag(header["space directions"])[::-1].tolist()


def get_pddca_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PDDCA dataset and convert it to hdf5 volumes.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the preprocessed hdf5 volumes.
    """
    data_dir = os.path.join(path, "preprocessed")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    raw_dir = os.path.join(path, "raw")
    for url, checksum in zip(URLS, CHECKSUMS):
        zip_path = os.path.join(path, os.path.basename(url))
        util.download_source(path=zip_path, url=url, download=download, checksum=checksum)
        util.unzip(zip_path=zip_path, dst=raw_dir)

    case_dirs = natsorted(glob(os.path.join(raw_dir, "0522c*")))
    assert len(case_dirs) == 48, f"Expected 48 cases, found {len(case_dirs)}."

    os.makedirs(data_dir, exist_ok=True)
    for case_dir in tqdm(case_dirs, desc="Converting PDDCA volumes to hdf5"):
        _convert_case(case_dir, os.path.join(data_dir, f"{os.path.basename(case_dir)}.h5"))

    return data_dir


def get_pddca_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "train_additional", "test_offsite", "test_onsite"]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the PDDCA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. One of 'train', 'train_additional', 'test_offsite' or 'test_onsite'.
            By default, all volumes are returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 volumes with image and label data.
    """
    data_dir = get_pddca_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "0522c*.h5")))

    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"'{split}' is not a valid split. Choose one of {list(SPLITS)}.")
        lower, upper = SPLITS[split]
        case_ids = [int(os.path.basename(p).replace("0522c", "").replace(".h5", "")) for p in volume_paths]
        volume_paths = [p for p, case_id in zip(volume_paths, case_ids) if lower <= case_id <= upper]

    assert len(volume_paths) > 0
    return volume_paths


def get_pddca_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Optional[Literal["train", "train_additional", "test_offsite", "test_onsite"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PDDCA dataset for organs-at-risk segmentation in head and neck CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. One of 'train', 'train_additional', 'test_offsite' or 'test_onsite'.
            By default, all volumes are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_pddca_paths(path, split, download)

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


def get_pddca_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Optional[Literal["train", "train_additional", "test_offsite", "test_onsite"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PDDCA dataloader for organs-at-risk segmentation in head and neck CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. One of 'train', 'train_additional', 'test_offsite' or 'test_onsite'.
            By default, all volumes are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pddca_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
