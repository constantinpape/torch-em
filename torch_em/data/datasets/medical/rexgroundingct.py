"""ReXGroundingCT links free-text chest radiology findings to pixel-level 3D lesion / finding
segmentations in non-contrast chest CT scans.

The dataset re-uses the CT volumes from CT-RATE (https://doi.org/10.48550/arXiv.2403.17834) and adds
segmentation masks for 8,028 findings across 14 abnormality categories in 3,142 CT scans: 2,992 scans
for training, 50 for public validation, and 100 held out privately for the MICCAI 2026 challenge
leaderboard. Each raw mask file is a 4D volume of shape (finding_category, X, Y, Z), one channel per
abnormality category present in that scan; within a channel, distinct positive values label distinct
entities of that finding (see `anatomical_cot.json` / `dataset.json` on the dataset repository for the
full per-finding metadata). This loader merges all categories/entities of a scan into a single 3D
instance segmentation volume with a globally unique instance id per (category, entity) pair.

NOTE: The masks are hosted at https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT, but the CT
volumes themselves are NOT included in that repository. They have to be fetched separately from CT-RATE
at https://huggingface.co/datasets/ibrahimhamamci/CT-RATE, using the matching case names (e.g. the mask
'segmentations/train_10000_a_1.nii.gz' corresponds to the CT-RATE volume at
'dataset/train_fixed/train_10000/train_10000_a/train_10000_a_1.nii.gz'). Both repositories are gated on
HuggingFace: visiting the dataset pages and accepting the license terms while logged in (self-service,
no manual review) is required before downloading with a HuggingFace access token.

NOTE: This loader only exposes the paired CT volume and finding-instance mask as a plain segmentation
dataset. It does not parse or expose the free-text finding descriptions / categories from
'dataset.json' or 'reports_dataset.json'; use `get_rexgroundingct_metadata` to load the raw per-case
metadata dictionary from 'dataset.json' if the associated text is needed.

The masks are licensed under CC BY-NC-SA 4.0. This dataset is from the publication
https://doi.org/10.48550/arXiv.2507.22030. Please cite it if you use this dataset in your research.
"""

import os
import json
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional, Dict, Any

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


REXGROUNDINGCT_REPO = "rajpurkarlab/ReXGroundingCT"
CT_RATE_REPO = "ibrahimhamamci/CT-RATE"

SPLITS = ["train", "val"]

SPLIT_TAGS = {"train": "train", "val": "valid"}
"""Mapping from the split name used by this module to the case name prefix used on the dataset repos."""


def _ct_rate_path(name: str) -> str:
    # e.g. 'train_10000_a_1' -> 'dataset/train_fixed/train_10000/train_10000_a/train_10000_a_1.nii.gz'
    parts = name.split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected ReXGroundingCT case name format: '{name}'")
    tag, case_id, letter, _idx = parts
    case_dir = f"{tag}_{case_id}"
    recon_dir = f"{tag}_{case_id}_{letter}"
    return f"dataset/{tag}_fixed/{case_dir}/{recon_dir}/{name}.nii.gz"


def _case_names_for_split(dataset_json: Dict[str, Any], split: str) -> List[str]:
    tag = SPLIT_TAGS[split]

    if split in dataset_json and isinstance(dataset_json[split], (list, dict)):
        entries = dataset_json[split]
        names = list(entries.keys()) if isinstance(entries, dict) else [e["name"] for e in entries]
    elif all(isinstance(v, dict) and "name" in v for v in dataset_json.values()):
        names = [v["name"] for v in dataset_json.values() if v["name"].startswith(f"{tag}_")]
    elif isinstance(dataset_json, dict) and all(
        k.startswith((SPLIT_TAGS["train"], SPLIT_TAGS["val"])) for k in dataset_json
    ):
        names = [k for k in dataset_json if k.startswith(f"{tag}_")]
    else:
        raise RuntimeError(
            "Could not determine the case names for the requested split from 'dataset.json'. The schema of "
            "this file could not be verified ahead of time because the dataset repository is gated; please "
            "inspect the downloaded 'dataset.json' and adjust '_case_names_for_split' accordingly."
        )

    names = [n[:-len(".nii.gz")] if n.endswith(".nii.gz") else n for n in names]
    return natsorted(set(names))


def get_rexgroundingct_metadata(path: Union[os.PathLike, str], download: bool = False) -> Dict[str, Any]:
    """Load the per-case finding metadata (free-text descriptions, categories, entity counts) shipped
    alongside the ReXGroundingCT masks.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        The parsed contents of 'dataset.json'.
    """
    json_path = os.path.join(path, "dataset.json")
    if not os.path.exists(json_path):
        if not download:
            raise RuntimeError(f"Cannot find the data at '{json_path}', but download was set to False.")
        _download_masks(path, [], download=True, only_json=True)

    with open(json_path, "r") as f:
        return json.load(f)


def _download_masks(path, case_names, download, only_json=False):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("'huggingface_hub' is required to download ReXGroundingCT. Install it via conda/pip.")

    os.makedirs(path, exist_ok=True)
    json_path = os.path.join(path, "dataset.json")

    if not os.path.exists(json_path):
        if not download:
            raise RuntimeError(f"Cannot find the data at '{json_path}', but download was set to False.")
        snapshot_download(repo_id=REXGROUNDINGCT_REPO, repo_type="dataset", local_dir=path, allow_patterns="*.json")

    if only_json:
        return

    missing = [n for n in case_names if not os.path.exists(os.path.join(path, "segmentations", f"{n}.nii.gz"))]
    if missing:
        if not download:
            raise RuntimeError(f"Cannot find {len(missing)} mask(s) at '{path}', but download was set to False.")
        patterns = [f"segmentations/{n}.nii.gz" for n in missing]
        snapshot_download(repo_id=REXGROUNDINGCT_REPO, repo_type="dataset", local_dir=path, allow_patterns=patterns)


def _download_volumes(path, case_names, download):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("'huggingface_hub' is required to download CT-RATE. Install it via conda/pip.")

    image_dir = os.path.join(path, "images")
    os.makedirs(image_dir, exist_ok=True)

    missing, remote_paths = [], {}
    for name in case_names:
        remote_path = _ct_rate_path(name)
        remote_paths[name] = remote_path
        if not os.path.exists(os.path.join(image_dir, os.path.basename(remote_path))):
            missing.append(remote_path)

    if missing:
        if not download:
            raise RuntimeError(f"Cannot find {len(missing)} CT volume(s) at '{image_dir}', but download was False.")
        snapshot_download(repo_id=CT_RATE_REPO, repo_type="dataset", local_dir=path, allow_patterns=missing)
        for remote_path in missing:
            src = os.path.join(path, remote_path)
            dst = os.path.join(image_dir, os.path.basename(remote_path))
            if os.path.exists(src) and not os.path.exists(dst):
                os.rename(src, dst)

    return image_dir, remote_paths


def _merge_instance_mask(mask: np.ndarray) -> np.ndarray:
    # 'mask' has shape (finding_category, X, Y, Z). Each channel's distinct positive values label
    # distinct entities of that finding category. Assign a globally unique instance id to every
    # (category, entity) pair across all channels.
    merged = np.zeros(mask.shape[1:], dtype="uint16")
    next_id = 1
    for channel in mask:
        for value in np.unique(channel):
            if value == 0:
                continue
            merged[channel == value] = next_id
            next_id += 1
    return merged


def _merge_masks(mask_dir: str, merged_dir: str, case_names: List[str]) -> None:
    import nibabel as nib

    os.makedirs(merged_dir, exist_ok=True)
    for name in case_names:
        dst = os.path.join(merged_dir, f"{name}.nii.gz")
        if os.path.exists(dst):
            continue
        src = os.path.join(mask_dir, f"{name}.nii.gz")
        image = nib.load(src)
        merged = _merge_instance_mask(np.asarray(image.dataobj))
        nib.save(nib.Nifti1Image(merged, image.affine), dst)


def get_rexgroundingct_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "val"],
    max_cases: Optional[int] = None,
    download: bool = False,
) -> Tuple[str, str]:
    """Download the ReXGroundingCT masks and the matching CT-RATE volumes.

    Both HuggingFace repositories are gated (self-service: accept the license on the dataset page while
    logged in, then pass a HuggingFace access token, e.g. via the `HF_TOKEN` environment variable).

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' (2,992 cases) or 'val' (public validation).
        max_cases: The maximum number of cases to download, taken in order. By default all cases of the
            requested split are downloaded.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the CT volumes.
        Filepath to the folder with the merged instance segmentation masks.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Please choose one of {SPLITS}.")

    mask_dir = os.path.join(path, "segmentations")
    merged_dir = os.path.join(path, "segmentations_merged")
    dataset_json = get_rexgroundingct_metadata(path, download)
    case_names = _case_names_for_split(dataset_json, split)
    if max_cases is not None:
        case_names = case_names[:max_cases]

    _download_masks(path, case_names, download)
    image_dir, _ = _download_volumes(path, case_names, download)
    _merge_masks(mask_dir, merged_dir, case_names)

    return image_dir, merged_dir


def get_rexgroundingct_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val"],
    max_cases: Optional[int] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the ReXGroundingCT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train' (2,992 cases) or 'val' (public validation).
        max_cases: The maximum number of cases to use. See `get_rexgroundingct_data` for details.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the CT volumes.
        List of filepaths for the finding masks.
    """
    image_dir, mask_dir = get_rexgroundingct_data(path, split, max_cases, download)

    label_paths = natsorted(glob(os.path.join(mask_dir, "*.nii.gz")))
    if max_cases is not None:
        label_paths = label_paths[:max_cases]

    raw_paths = [os.path.join(image_dir, os.path.basename(p)) for p in label_paths]
    if len(raw_paths) == 0 or not all(os.path.exists(p) for p in raw_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return raw_paths, label_paths


def get_rexgroundingct_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val"],
    max_cases: Optional[int] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ReXGroundingCT dataset for lesion / finding segmentation in chest CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' (2,992 cases) or 'val' (public validation).
        max_cases: The maximum number of cases to use. See `get_rexgroundingct_data` for details.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_rexgroundingct_paths(path, split, max_cases, download)

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


def get_rexgroundingct_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val"],
    max_cases: Optional[int] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ReXGroundingCT dataloader for lesion / finding segmentation in chest CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' (2,992 cases) or 'val' (public validation).
        max_cases: The maximum number of cases to use. See `get_rexgroundingct_data` for details.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_rexgroundingct_dataset(path, patch_shape, split, max_cases, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
