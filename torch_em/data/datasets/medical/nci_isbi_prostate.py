"""The NCI-ISBI 2013 Prostate dataset contains annotations for prostate zone segmentation in T2-weighted MRI.

The data comes from the 'NCI-ISBI 2013 Challenge: Automated Segmentation of Prostate Structures' and consists of
80 T2-weighted MRI of the prostate, 40 acquired at 3T with a surface coil (the TCIA collection 'Prostate-3T') and
40 acquired at 1.5T with an endorectal coil (the TCIA collection 'PROSTATE-DIAGNOSIS'). The official challenge split
is exposed via the 'split' argument: 'train' (60 volumes), 'leaderboard' (10 volumes) and 'test' (10 volumes).

The MRI are distributed as DICOM series on TCIA and the manual segmentations as NRRD files, which are converted and
stored together in hdf5 files by this module. The label ids are: 0 = background, 1 = peripheral zone,
2 = central gland.

NOTE: This requires the pydicom and pynrrd python packages.

The dataset is located at https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013/.

This dataset is from the publication https://doi.org/10.7937/K9/TCIA.2015.zF0vlOPv.
Please cite it if you use this dataset in your research.
"""

import os
import csv
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/"

URLS = {
    "train": {
        "images": BASE_URL + "ISBI-Prostate-Challenge-Training.tcia",
        "labels": BASE_URL + "NCI-ISBI-2013-Prostate-Challenge-Training.zip",
    },
    "leaderboard": {
        "images": BASE_URL + "ISBI-Prostate-Challenge-LeaderBoard.tcia",
        "labels": BASE_URL + "NCI-ISBI-2013-Prostate-Challenge-Leaderboard.zip",
    },
    "test": {
        "images": BASE_URL + "ISBI-Prostate-Challenge-Testing.tcia",
        "labels": BASE_URL + "NCI-ISBI-2013-Prostate-Challenge-Test.zip",
    },
}

# The DICOM series are downloaded individually from TCIA, so only the label archives have a checksum.
CHECKSUMS = {
    "train": "c3436559b474c60e78633ea98601241f39cb27e30fb9d48b1d29f2821bfbf047",
    "leaderboard": "011945fdd273b2f43f6ba2de8904ad8625800b439dff9d228810b645c65fa0be",
    "test": "53c2969c035974de47bcb2d3fa20c1342c99096faf2fc87a0097ed62e89d32e3",
}

LABEL_IDS = {"background": 0, "peripheral_zone": 1, "central_gland": 2}

SPLITS = {"train": "Training", "leaderboard": "Leaderboard", "test": "Test"}

N_VOLUMES = {"train": 60, "leaderboard": 10, "test": 10}


def _get_dicom_affine(geometry):
    """Build the affine that maps the voxel indices (z, y, x) of a DICOM volume to patient coordinates."""
    origin = geometry["origin"]
    if len(origin) > 1:
        slice_step = (origin[-1] - origin[0]) / (len(origin) - 1)
    else:
        slice_step = np.cross(geometry["row_direction"], geometry["column_direction"])

    affine = np.eye(4)
    affine[:3, 0] = slice_step
    affine[:3, 1] = geometry["column_direction"] * geometry["spacing"][0]
    affine[:3, 2] = geometry["row_direction"] * geometry["spacing"][1]
    affine[:3, 3] = origin[0]
    return affine


def _load_nrrd_on_dicom_grid(nrrd_path, geometry, shape):
    """Load a NRRD label volume and resample it onto the voxel grid of the DICOM series.

    Most segmentations are stored on exactly the grid of the DICOM series, but with a different axis order,
    while some (mostly in the 'PROSTATE-DIAGNOSIS' part) are stored on a grid with a different slice spacing.
    Both cases are handled by mapping each voxel of the DICOM grid to patient coordinates via the DICOM affine
    and back to a NRRD voxel index via the inverse of the NRRD affine, i.e. by nearest neighbor resampling.
    """
    import nrrd

    labels, header = nrrd.read(nrrd_path)
    directions = np.asarray(header["space directions"], dtype="float64")
    origin = np.asarray(header["space origin"], dtype="float64")
    if header.get("space", "").lower().startswith("right-anterior"):  # Convert RAS to the LPS used by DICOM.
        directions = directions * np.array([-1.0, -1.0, 1.0])
        origin = origin * np.array([-1.0, -1.0, 1.0])

    nrrd_affine = np.eye(4)
    nrrd_affine[:3, :3] = directions.T  # The rows of 'space directions' are the directions of the NRRD axes.
    nrrd_affine[:3, 3] = origin

    to_nrrd_index = np.linalg.inv(nrrd_affine) @ _get_dicom_affine(geometry)
    resampled = np.zeros(shape, dtype="uint8")
    yy, xx = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    for z in range(shape[0]):
        target_indices = np.stack([np.full(yy.size, z), yy.ravel(), xx.ravel(), np.ones(yy.size)])
        indices = np.round(to_nrrd_index[:3] @ target_indices).astype("int")
        valid = np.all((indices >= 0) & (indices < np.array(labels.shape)[:, None]), axis=0)
        resampled[z].flat[valid] = labels[tuple(indices[:, valid])]

    return resampled


def _preprocess_inputs(dicom_dir, csv_path, label_dir, preprocessed_dir):
    import h5py

    with open(csv_path, "r") as f:
        series_uids = {row["Subject ID"]: row["Series UID"] for row in csv.DictReader(f)}

    label_paths = {}
    for label_path in natsorted(glob(os.path.join(label_dir, "*", "*.nrrd"))):
        subject_id = os.path.basename(label_path).split(".")[0]
        # Some label files carry a suffix, e.g. 'ProstateDx-01-0006_correctedLabels.nrrd'.
        label_paths[subject_id.split("_")[0]] = label_path

    os.makedirs(preprocessed_dir, exist_ok=True)
    for subject_id, label_path in tqdm(sorted(label_paths.items()), desc="Preprocess NCI-ISBI 2013 Prostate"):
        volume_path = os.path.join(preprocessed_dir, f"{subject_id}.h5")
        if os.path.exists(volume_path):
            continue

        raw, geometry = util.load_dicom_series(os.path.join(dicom_dir, series_uids[subject_id]))
        raw = np.round(raw).astype("int16")
        labels = _load_nrrd_on_dicom_grid(label_path, geometry, raw.shape)

        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)


def get_nci_isbi_prostate_data(
    path: Union[os.PathLike, str], split: Literal["train", "leaderboard", "test"], download: bool = False
) -> str:
    """Download the NCI-ISBI 2013 Prostate dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train', 'leaderboard' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Please choose one of {list(SPLITS.keys())}.")

    preprocessed_dir = os.path.join(path, "preprocessed", split)
    if len(glob(os.path.join(preprocessed_dir, "*.h5"))) == N_VOLUMES[split]:
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    label_dir = os.path.join(path, "labels", split)
    if not os.path.exists(label_dir):
        zip_path = os.path.join(path, f"{SPLITS[split]}.zip")
        util.download_source(path=zip_path, url=URLS[split]["labels"], download=download, checksum=CHECKSUMS[split])
        util.unzip(zip_path=zip_path, dst=label_dir)

    dicom_dir = os.path.join(path, "dicom", split)
    csv_filename = os.path.join(path, f"{split}_metadata")
    if not os.path.exists(f"{csv_filename}.csv"):
        util.download_source_tcia(
            path=os.path.join(path, f"{split}.tcia"),
            url=URLS[split]["images"],
            dst=dicom_dir,
            csv_filename=csv_filename,
            download=download,
        )

    _preprocess_inputs(dicom_dir, f"{csv_filename}.csv", label_dir, preprocessed_dir)
    return preprocessed_dir


def get_nci_isbi_prostate_paths(
    path: Union[os.PathLike, str], split: Literal["train", "leaderboard", "test"], download: bool = False
) -> List[str]:
    """Get paths to the NCI-ISBI 2013 Prostate data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train', 'leaderboard' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_nci_isbi_prostate_data(path, split, download)
    return natsorted(glob(os.path.join(data_dir, "*.h5")))


def get_nci_isbi_prostate_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "leaderboard", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the NCI-ISBI 2013 Prostate dataset for prostate zone segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'leaderboard' or 'test'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_nci_isbi_prostate_paths(path, split, download)

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


def get_nci_isbi_prostate_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "leaderboard", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the NCI-ISBI 2013 Prostate dataloader for prostate zone segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'leaderboard' or 'test'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_nci_isbi_prostate_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
