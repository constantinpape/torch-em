"""The BraTS 2024 dataset contains annotations for the sub-regions of post-treatment adult diffuse
glioma in multi-modal brain MRI.

It is the adult glioma segmentation task of the 2024 Brain Tumor Segmentation (BraTS) challenge
(https://www.synapse.org/Synapse:syn53708126). Unlike the BraTS 2023 release (see `brats.py`), this is an
entirely new dataset of exclusively post-treatment studies, with 1621 training studies. Each study provides
four co-registered, skull-stripped and interpolated sequences, which can be selected with the 'modality'
argument: a native T1-weighted scan ('t1n'), a post-contrast T1-weighted scan ('t1c'), a T2-weighted scan
('t2w') and a T2 FLAIR scan ('t2f').

The label ids are described in `LABEL_IDS`: 0 = background, 1 = non-enhancing tumor core (NETC), 2 =
surrounding non-enhancing FLAIR hyperintensity (SNFH), 3 = enhancing tissue (ET), 4 = resection cavity (RC).
NOTE: These ids differ from the BraTS 2023 (and earlier) releases, which do not have a resection cavity
class and instead use the id 1 for the necrotic tumor core.

Evaluation is not done on the sub-regions themselves, but on the nested regions that they form, which can be
selected with the 'region' argument (see `REGIONS`): the whole tumor (the union of the non-enhancing tumor
core, the FLAIR hyperintensity and the enhancing tissue, excluding the resection cavity), the tumor core
(the union of the non-enhancing tumor core and the enhancing tissue) and the enhancing tissue. The
individual sub-regions, including the resection cavity, can also be selected as a binary target with this
argument. By default, the sub-region ids are returned as they are. This follows the official evaluation
script at https://github.com/rachitsaluja/BraTS-2024-Metrics.

NOTE: The official data at https://www.synapse.org/Synapse:syn53708126 is only handed out to registered
participants, so this module downloads a public mirror of the BraTS 2024 adult glioma training set at
https://huggingface.co/datasets/Spirit-26/BraTS-2024-Complete. If the official release is extracted into
the folder passed as 'path', so that files such as
'<path>/**/BraTS-GLI-00005-100/BraTS-GLI-00005-100-t2f.nii.gz' exist, it is used instead of the mirror.

The scans are used as nifti volumes directly (the key is 'data'). They are loaded with the axis order
reversed with respect to the nifti file, i.e. (Z, Y, X), so that a 2d patch shape selects axial slices.

This dataset is from the publication https://doi.org/10.48550/arXiv.2405.18368.
Please cite it if you use this dataset in your research.
"""

import os
import json
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


FOLDER_NAME = "BraTS-GLI"

URL_BASE = f"https://huggingface.co/datasets/Spirit-26/BraTS-2024-Complete/resolve/main/{FOLDER_NAME}/train"

API_URL = f"https://huggingface.co/api/datasets/Spirit-26/BraTS-2024-Complete/tree/main/{FOLDER_NAME}/train"

LABEL_IDS = {
    "background": 0,
    "non_enhancing_tumor_core": 1,
    "surrounding_flair_hyperintensity": 2,
    "enhancing_tissue": 3,
    "resection_cavity": 4,
}

# The nested tumor regions that the challenge evaluates, and the sub-regions they are made of.
REGIONS = {
    "whole_tumor": (1, 2, 3),
    "tumor_core": (1, 3),
    "enhancing_tissue": (3,),
    "surrounding_flair_hyperintensity": (2,),
    "non_enhancing_tumor_core": (1,),
    "resection_cavity": (4,),
}

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]

N_SUBJECTS = 1621

N_RETRIES = 5


class RegionTransform:
    """Transform the BraTS 2024 sub-region ids into a binary mask for one of the tumor regions.

    Args:
        region: The name of the tumor region, see `REGIONS`.
    """
    def __init__(self, region: str):
        self.region = region

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        """Apply the transform.

        Args:
            labels: The sub-region ids.

        Returns:
            The binary mask of the tumor region.
        """
        return np.isin(labels, REGIONS[self.region]).astype("uint8")


def _get_subject_ids(path, download):
    """List the studies of the mirror via the huggingface API and cache the listing next to the data."""
    listing_path = os.path.join(path, "subject_ids.json")
    if os.path.exists(listing_path):
        with open(listing_path, "r") as f:
            return json.load(f)

    if not download:
        raise RuntimeError(f"Cannot find the data at '{path}', but download was set to False.")

    import requests

    subject_ids, cursor = [], None
    while True:
        params = {"limit": 1000}
        if cursor is not None:
            params["cursor"] = cursor

        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        subject_ids.extend(os.path.basename(entry["path"]) for entry in response.json())

        link = response.headers.get("Link", "")
        if 'rel="next"' not in link:
            break
        cursor = link.split("cursor=")[1].split("&")[0].split(">")[0]

    subject_ids = natsorted(subject_ids)
    assert len(subject_ids) == N_SUBJECTS, f"Expected {N_SUBJECTS} studies in the mirror, got {len(subject_ids)}."

    with open(listing_path, "w") as f:
        json.dump(subject_ids, f)

    return subject_ids


def _find_data(path, modality):
    """Find the studies on disk, both for the official release and for the mirror downloaded by this module."""
    pattern = os.path.join(path, "**", "BraTS-GLI-*", f"BraTS-GLI-*-{modality}.nii.gz")
    raw_paths = natsorted(glob(pattern, recursive=True))
    label_paths = [p.replace(f"-{modality}.nii.gz", "-seg.nii.gz") for p in raw_paths]

    keep = [i for i, p in enumerate(label_paths) if os.path.exists(p)]
    return [raw_paths[i] for i in keep], [label_paths[i] for i in keep]


def _download_volumes(path, modality, download):
    raw_paths, label_paths = [], []
    for subject_id in tqdm(_get_subject_ids(path, download), desc="Downloading the BraTS 2024 studies"):
        subject_dir = os.path.join(path, FOLDER_NAME, subject_id)
        os.makedirs(subject_dir, exist_ok=True)

        for suffix in [modality, "seg"]:
            fname = f"{subject_id}-{suffix}.nii.gz"
            fpath = os.path.join(subject_dir, fname)
            # The mirror is fetched file by file, so a transient error is retried instead of failing the download.
            for attempt in range(N_RETRIES):
                try:
                    util.download_source(path=fpath, url=f"{URL_BASE}/{subject_id}/{fname}", download=download)
                    break
                except Exception:
                    if attempt == N_RETRIES - 1:
                        raise

            (label_paths if suffix == "seg" else raw_paths).append(fpath)

    return raw_paths, label_paths


def get_brats24_data(
    path: Union[os.PathLike, str],
    modality: Literal["t1n", "t1c", "t2w", "t2f"] = "t2f",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Download the BraTS 2024 post-treatment adult glioma dataset.

    Only the requested modality and the annotations are downloaded, since the studies are fetched study
    by study from the mirror.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The MRI sequence. Either 't1n', 't1c', 't2w' or 't2f'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Please choose one of {MODALITIES}.")

    os.makedirs(path, exist_ok=True)

    raw_paths, label_paths = _find_data(path, modality)
    if len(raw_paths) == N_SUBJECTS:
        return raw_paths, label_paths

    return _download_volumes(path, modality, download)


def get_brats24_paths(
    path: Union[os.PathLike, str],
    modality: Literal["t1n", "t1c", "t2w", "t2f"] = "t2f",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the BraTS 2024 post-treatment adult glioma data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The MRI sequence. Either 't1n', 't1c', 't2w' or 't2f'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    raw_paths, label_paths = get_brats24_data(path, modality, download)
    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0, f"Could not find the studies in '{path}'."
    return raw_paths, label_paths


def get_brats24_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Literal["t1n", "t1c", "t2w", "t2f"] = "t2f",
    region: Optional[Literal[
        "whole_tumor", "tumor_core", "enhancing_tissue",
        "surrounding_flair_hyperintensity", "non_enhancing_tumor_core", "resection_cavity",
    ]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BraTS 2024 post-treatment adult glioma dataset for brain tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The MRI sequence. Either 't1n', 't1c', 't2w' or 't2f'.
        region: The tumor region to use as a binary target, see `REGIONS`. If None, the sub-region ids are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if region is not None and region not in REGIONS:
        raise ValueError(f"'{region}' is not a valid region. Please choose one of {list(REGIONS.keys())}.")

    raw_paths, label_paths = get_brats24_paths(path, modality, download)

    if region is not None:
        kwargs = util.update_kwargs(kwargs, "label_transform", RegionTransform(region))

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


def get_brats24_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Literal["t1n", "t1c", "t2w", "t2f"] = "t2f",
    region: Optional[Literal[
        "whole_tumor", "tumor_core", "enhancing_tissue",
        "surrounding_flair_hyperintensity", "non_enhancing_tumor_core", "resection_cavity",
    ]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BraTS 2024 post-treatment adult glioma dataloader for brain tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The MRI sequence. Either 't1n', 't1c', 't2w' or 't2f'.
        region: The tumor region to use as a binary target, see `REGIONS`. If None, the sub-region ids are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_brats24_dataset(path, patch_shape, modality, region, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
