"""The ATLAS Stroke dataset contains annotations for stroke lesion segmentation in T1-weighted brain MRI.

It is version 2.0 of ATLAS (Anatomical Tracings of Lesions After Stroke, https://atlas.grand-challenge.org),
a collection of 1271 T1-weighted MRI scans of chronic stroke patients pooled from 44 research cohorts, in
which the lesions were traced manually. The release is split into a training set of 655 scans with public
lesion masks, a test set of 300 scans whose masks are withheld for the challenge evaluation, and a
generalizability set of 316 scans whose scans and masks are both withheld. This module therefore provides
the 655 annotated scans of the training set.

The scans are defaced, intensity normalized and registered to the MNI-152 template
('space-MNI152NLin2009aSym'), so all of them have a shape of (197, 233, 189) at 1 mm isotropic resolution.

The annotations are binary, see `LABEL_IDS`: 1 = stroke lesion.

NOTE: The official release at https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html is only handed out
after agreeing to the terms of use in a request form, and is distributed as an encrypted archive, so it
cannot be downloaded automatically. This module downloads a public mirror of the training set at
https://huggingface.co/datasets/jayzzzzz0134/atlas-stroke instead. If the official BIDS release is extracted
into the folder passed as 'path', so that files such as
'<path>/**/sub-r001s001/ses-1/anat/sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz' and the matching
'..._label-L_desc-T1lesion_mask.nii.gz' exist, it is used instead of the mirror.

The scans are used as nifti volumes directly (the key is 'data'). They are loaded with the axis order
reversed with respect to the nifti file, i.e. (Z, Y, X), so that a 2d patch shape selects axial slices.

This dataset is from the publication https://doi.org/10.1038/s41597-022-01401-7.
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

from .. import util


URL_BASE = "https://huggingface.co/datasets/jayzzzzz0134/atlas-stroke/resolve/main"

API_URL = "https://huggingface.co/api/datasets/jayzzzzz0134/atlas-stroke/tree/main/masks"

LABEL_IDS = {"background": 0, "stroke_lesion": 1}

N_SUBJECTS = 655

N_RETRIES = 5


def _get_subject_ids(path, download):
    """List the subjects of the mirror via the huggingface API and cache the listing next to the data."""
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
        subject_ids.extend(
            os.path.basename(entry["path"]).replace("_lesion_mask.nii.gz", "") for entry in response.json()
        )

        link = response.headers.get("Link", "")
        if 'rel="next"' not in link:
            break
        cursor = link.split("cursor=")[1].split("&")[0].split(">")[0]

    subject_ids = natsorted(subject_ids)
    assert len(subject_ids) == N_SUBJECTS, f"Expected {N_SUBJECTS} subjects in the mirror, got {len(subject_ids)}."

    with open(listing_path, "w") as f:
        json.dump(subject_ids, f)

    return subject_ids


def _find_official_data(path):
    """Find the scans of the official BIDS release, in case it was downloaded manually."""
    pattern = os.path.join(path, "**", "sub-*_space-MNI152NLin2009aSym_T1w.nii.gz")
    raw_paths = natsorted(glob(pattern, recursive=True))
    label_paths = [p.replace("_T1w.nii.gz", "_label-L_desc-T1lesion_mask.nii.gz") for p in raw_paths]

    keep = [i for i, p in enumerate(label_paths) if os.path.exists(p)]
    return [raw_paths[i] for i in keep], [label_paths[i] for i in keep]


def _download_volumes(path, download):
    for folder in ["images", "masks"]:
        os.makedirs(os.path.join(path, folder), exist_ok=True)

    raw_paths, label_paths = [], []
    for subject_id in tqdm(_get_subject_ids(path, download), desc="Downloading the ATLAS Stroke scans"):
        for folder, fname in [("images", f"{subject_id}_T1w.nii.gz"), ("masks", f"{subject_id}_lesion_mask.nii.gz")]:
            fpath = os.path.join(path, folder, fname)
            # The mirror is fetched file by file, so a transient error is retried instead of failing the download.
            for attempt in range(N_RETRIES):
                try:
                    util.download_source(path=fpath, url=f"{URL_BASE}/{folder}/{fname}", download=download)
                    break
                except Exception:
                    if attempt == N_RETRIES - 1:
                        raise

            (raw_paths if folder == "images" else label_paths).append(fpath)

    return raw_paths, label_paths


def get_atlas_stroke_data(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Download the ATLAS v2.0 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    os.makedirs(path, exist_ok=True)

    raw_paths, label_paths = _find_official_data(path)
    if len(raw_paths) > 0:
        return raw_paths, label_paths

    return _download_volumes(path, download)


def get_atlas_stroke_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the ATLAS v2.0 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    raw_paths, label_paths = get_atlas_stroke_data(path, download)
    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0, f"Could not find the scans in '{path}'."
    return raw_paths, label_paths


def get_atlas_stroke_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ATLAS v2.0 dataset for stroke lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_atlas_stroke_paths(path, download)

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


def get_atlas_stroke_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ATLAS v2.0 dataloader for stroke lesion segmentation.

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
    dataset = get_atlas_stroke_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
