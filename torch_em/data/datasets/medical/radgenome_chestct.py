"""RadGenome-Chest CT extends CT-RATE (https://huggingface.co/datasets/ibrahimhamamci/CT-RATE, 25692 non-contrast
chest CT volumes from 20000 patients) with organ-level segmentation masks for 197 anatomical structures, as well
as grounded reports and VQA pairs (text-only; not covered by this loader).

Unlike CT-RATE itself, which requires accepting a gated access agreement on HuggingFace before it can be
downloaded, the RadGenome-ChestCT repository (https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT) is
NOT access-gated (verified via the HuggingFace API and by downloading files without any token). It bundles
its own already-preprocessed copy of the CT-RATE volumes (resampled to (3, 1, 1) mm spacing, cropped to the
foreground and stored with an identity affine, see 'processed_code/preprocess_ctrate_valid.py' in the
repository) under 'dataset/<split>_preprocessed/', where '<split>' is 'train' or 'valid'. This loader uses
these preprocessed volumes directly as the raw data, so the original (gated) CT-RATE repository is not needed.

The per-structure binary segmentation masks are stored separately, keyed by the same volume name as the
preprocessed CT (eg. the CT 'dataset/valid_preprocessed/valid_1/valid_1a/valid_1_a_1.nii.gz' corresponds to the
mask folder 'seg_valid_1_a_1/', containing one binary '<structure_name>.nii.gz' file per structure). For the
validation split, all masks are bundled in a single 'dataset/valid_anatomy_mask.tar.gz' (~10.5 GB); for the
much larger training split, they are split into 5 parts, 'dataset/train_anatomy_mask_a{a,b,c,d,e}'
(~13-16 GB each, ~74 GB combined), which have to be concatenated into one tar.gz before any file inside can be
extracted. Because of this, downloading masks for even a single training-split case requires downloading the
full ~74 GB of split archive parts; there is no way around this with the way the data is packaged. The
validation split masks (~10.5 GB) are comparatively cheap, so this loader defaults to `split="validation"`.
Use `max_cases`/`case_ids` to restrict how many cases are used (and thus how many preprocessed CT files are
downloaded); the full mask archive of the chosen split is always downloaded once, regardless of the subset
size, since the masks cannot be fetched per case.

Inspecting the actual downloaded validation-split archive shows that every one of its 1564 cases has exactly
210 per-structure mask files (empty structures are still stored as empty masks, as for AbdomenAtlas 3.0, see
`abdomen_atlas_3.py`). These 210 structure names are stored in `CLASS_NAMES` (ids 1-210 via `CLASS_IDS`, sorted
alphabetically); they do not exactly match the paper's "197 categories" figure, most likely because that figure
groups some of the numbered/lateralized variants together (eg. the 24 per-vertebra and per-rib masks, or the
left/right variants of paired organs). `CLASS_NAMES` was derived empirically from the downloaded validation
split and is used as the canonical vocabulary for both splits; if the training split contains additional
structure names not in this list, `merge_segmentations` skips them (a warning is printed) rather than failing.

The dataset is licensed under CC BY 4.0. Despite the license, using it still requires agreeing to CT-RATE's
usage terms (non-commercial, no redistribution, no re-identification attempts), as reproduced in the dataset
card at https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT.

This dataset is from the publication https://arxiv.org/abs/2404.16754 (Scientific Data 2025).
Please also cite CT-RATE (https://arxiv.org/abs/2403.17834) and CT-CLIP if you use this dataset in your
research.
"""

import os
import warnings
import subprocess
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Optional

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


REPO_ID = "RadGenome/RadGenome-ChestCT"

SPLIT_PREFIXES = {"train": "train", "validation": "valid"}

CLASS_NAMES = [
    "abdominal tissue", "adrenal gland", "aorta", "bone", "brachiocephalic trunk", "brachiocephalic vein",
    "breast", "bronchie", "buccal mucosa", "carotid artery", "caudate lobe", "celiac trunk",
    "cervical esophagus", "cervical vertebrae", "cervical vertebrae 1 (c1)", "cervical vertebrae 2 (c2)",
    "cervical vertebrae 3 (c3)", "cervical vertebrae 4 (c4)", "cervical vertebrae 5 (c5)",
    "cervical vertebrae 6 (c6)", "cervical vertebrae 7 (c7)", "clavicle", "colon", "common carotid artery",
    "costal cartilage", "cricopharyngeal inlet", "duodenum", "esophagus", "eustachian tube bone", "femur",
    "gallbladder", "head of femur", "head of left femur", "head of right femur", "heart",
    "heart ascending aorta", "heart atrium", "heart tissue", "heart ventricle", "humerus", "iliac artery",
    "iliac vena", "inferior vena cava", "internal carotid artery", "internal jugular vein", "intestine",
    "kidney", "kidney cyst", "kidney tumor", "larynx", "larynx glottis", "larynx supraglottis",
    "left adrenal gland", "left auricle of heart", "left brachiocephalic vein", "left breast",
    "left carotid artery", "left clavicle", "left common carotid artery", "left eustachian tube bone",
    "left femur", "left heart atrium", "left heart ventricle", "left humerus", "left iliac artery",
    "left iliac vena", "left internal carotid artery", "left internal jugular vein", "left kidney",
    "left kidney cyst", "left lateral inferior segment of liver", "left lateral superior segment of liver",
    "left lobe of liver", "left lung", "left lung lower lobe", "left lung upper lobe",
    "left medial segment of liver", "left rib", "left rib 1", "left rib 10", "left rib 11", "left rib 12",
    "left rib 2", "left rib 3", "left rib 4", "left rib 5", "left rib 6", "left rib 7", "left rib 8",
    "left rib 9", "left scapula", "left subclavian artery", "left thyroid", "liver", "liver tumor",
    "liver vessel", "lumbar vertebrae", "lumbar vertebrae 1 (l1)", "lumbar vertebrae 2 (l2)",
    "lumbar vertebrae 3 (l3)", "lumbar vertebrae 4 (l4)", "lumbar vertebrae 5 (l5)", "lumbar vertebrae 6 (l6)",
    "lung", "lung effusion", "lung lower lobe", "lung nodule", "lung tumor", "lung upper lobe", "mandible",
    "manubrium of sternum", "mediastinal tissue", "muscle", "myocardium", "pancreas", "pancreas tumor",
    "portal vein and splenic vein", "prostate", "pulmonary artery", "pulmonary embolism", "pulmonary vein",
    "rectum", "renal artery", "renal vein", "rib", "rib 1", "rib 10", "rib 11", "rib 12", "rib 2", "rib 3",
    "rib 4", "rib 5", "rib 6", "rib 7", "rib 8", "rib 9", "rib cartilage", "right adrenal gland",
    "right anterior inferior segment of liver", "right anterior superior segment of liver",
    "right brachiocephalic vein", "right breast", "right carotid artery", "right clavicle",
    "right common carotid artery", "right eustachian tube bone", "right femur", "right heart atrium",
    "right heart ventricle", "right humerus", "right iliac artery", "right iliac vena",
    "right internal carotid artery", "right internal jugular vein", "right kidney", "right kidney cyst",
    "right lobe of liver", "right lung", "right lung lower lobe", "right lung middle lobe",
    "right lung upper lobe", "right posterior inferior segment of liver",
    "right posterior superior segment of liver", "right rib", "right rib 1", "right rib 10", "right rib 11",
    "right rib 12", "right rib 2", "right rib 3", "right rib 4", "right rib 5", "right rib 6", "right rib 7",
    "right rib 8", "right rib 9", "right scapula", "right subclavian artery", "right thyroid",
    "sacral vertebrae 1 (s1)", "scapula", "skin", "small bowel", "spinal canal", "spinal cord", "spleen",
    "sternum", "stomach", "subclavian artery", "superior vena cava", "thoracic cavity", "thoracic vertebrae",
    "thoracic vertebrae 1 (t1)", "thoracic vertebrae 10 (t10)", "thoracic vertebrae 11 (t11)",
    "thoracic vertebrae 12 (t12)", "thoracic vertebrae 2 (t2)", "thoracic vertebrae 3 (t3)",
    "thoracic vertebrae 4 (t4)", "thoracic vertebrae 5 (t5)", "thoracic vertebrae 6 (t6)",
    "thoracic vertebrae 7 (t7)", "thoracic vertebrae 8 (t8)", "thoracic vertebrae 9 (t9)", "thymus", "thyroid",
    "thyroid gland", "trachea", "vertebrae",
]
"""The 210 anatomical structure names found in the RadGenome-ChestCT masks (validation split), sorted
alphabetically. See the module docstring for how this relates to the paper's "197 categories" figure."""

CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
"""Mapping from structure name to its label id in the combined label volumes created by `merge_segmentations`."""


def _volume_name(relpath: str) -> str:
    return os.path.splitext(os.path.splitext(os.path.basename(relpath))[0])[0]


def _find_mask_dir(path: str, volume_name: str) -> Optional[str]:
    matches = glob(os.path.join(path, "**", f"seg_{volume_name}"), recursive=True)
    return matches[0] if matches else None


def _find_image_path(path: str, volume_name: str) -> Optional[str]:
    matches = glob(os.path.join(path, "**", f"{volume_name}.nii.gz"), recursive=True)
    matches = [m for m in matches if os.sep + "preprocessed" + os.sep in m or "preprocessed" in m]
    return matches[0] if matches else None


def merge_segmentations(mask_dir: str) -> str:
    """Merge the per-structure binary masks of one RadGenome-ChestCT case into a single semantic label volume.

    The merged volume is stored as 'combined_labels.nii.gz' in the mask folder. If it already exists,
    it is not recomputed.

    Args:
        mask_dir: The 'seg_<volume_name>' folder holding the per-structure masks.

    Returns:
        The filepath to the merged label volume.
    """
    import nibabel as nib

    label_path = os.path.join(mask_dir, "combined_labels.nii.gz")
    if os.path.exists(label_path):
        return label_path

    labels, affine = None, None
    for mask_path in natsorted(glob(os.path.join(mask_dir, "*.nii.gz"))):
        class_name = os.path.splitext(os.path.splitext(os.path.basename(mask_path))[0])[0]
        if class_name not in CLASS_IDS:
            warnings.warn(f"Skipping unknown structure '{class_name}' in '{mask_dir}'.")
            continue
        nifti = nib.load(mask_path)
        mask = np.asarray(nifti.dataobj) > 0
        if labels is None:
            labels, affine = np.zeros(mask.shape, dtype="uint8"), nifti.affine
        labels[mask] = CLASS_IDS[class_name]

    if labels is None:
        raise RuntimeError(f"Could not find any segmentation masks in '{mask_dir}'.")

    nib.save(nib.Nifti1Image(labels, affine), label_path)
    return label_path


def _concatenate_train_mask_shards(path: str) -> str:
    combined_path = os.path.join(path, "dataset", "train_anatomy_mask.tar.gz")
    if os.path.exists(combined_path):
        return combined_path

    part_paths = natsorted(glob(os.path.join(path, "dataset", "train_anatomy_mask_a?")))
    assert len(part_paths) == 5, f"Expected 5 'train_anatomy_mask_a*' parts, found {len(part_paths)} in '{path}'."

    print("Concatenating the 5 train anatomy mask shards into one archive (~74 GB). This can take a while.")
    with open(combined_path, "wb") as dst:
        subprocess.run(["cat"] + part_paths, stdout=dst, check=True)
    return combined_path


def get_radgenome_chestct_data(
    path: Union[os.PathLike, str],
    split: str = "validation",
    max_cases: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
    download: bool = False,
) -> List[str]:
    """Download the RadGenome-ChestCT dataset.

    The dataset is not gated on HuggingFace and does not require an access token. It is, however, very
    large: even the validation split requires downloading the full ~10.5 GB mask archive regardless of how
    many cases are requested (masks cannot be fetched per case), and the training split requires downloading
    ~74 GB of mask archive shards for the same reason. Use `max_cases`/`case_ids` to at least limit how many
    preprocessed CT volumes are downloaded; leave both at their default (None) to use the full split.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split, either 'train' or 'validation'.
        max_cases: The maximum number of cases to use, taken in natural sort order. Mutually exclusive
            with `case_ids`.
        case_ids: Explicit list of case / volume ids (eg. ['valid_1_a_1']) to use. Mutually exclusive
            with `max_cases`.
        download: Whether to download the data if it is not present.

    Returns:
        The volume names (eg. 'valid_1_a_1') of the available cases.
    """
    assert split in SPLIT_PREFIXES, f"'split' must be one of {list(SPLIT_PREFIXES)}, got '{split}'."
    assert max_cases is None or case_ids is None, "'max_cases' and 'case_ids' are mutually exclusive."
    prefix = SPLIT_PREFIXES[split]

    def _available_volumes():
        image_dir = os.path.join(path, "dataset", f"{prefix}_preprocessed")
        volumes = [
            _volume_name(p) for p in glob(os.path.join(image_dir, "*", "*", "*.nii.gz"))
            if _find_mask_dir(path, _volume_name(p)) is not None
        ]
        return natsorted(volumes)

    volumes = _available_volumes()
    if volumes:
        if max_cases is not None:
            volumes = volumes[:max_cases]
        elif case_ids is not None:
            volumes = [v for v in volumes if v in case_ids]
        if volumes:
            return volumes

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False")

    from huggingface_hub import HfApi, snapshot_download

    os.makedirs(path, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    image_files = [f for f in files if f.startswith(f"dataset/{prefix}_preprocessed/") and f.endswith(".nii.gz")]
    assert image_files, f"Could not find any preprocessed volumes for split '{split}' in the repository."
    volname_to_relpath = {_volume_name(f): f for f in image_files}
    all_volumes = natsorted(volname_to_relpath.keys())

    if case_ids is not None:
        target_volumes = [v for v in all_volumes if v in case_ids]
    elif max_cases is not None:
        target_volumes = all_volumes[:max_cases]
    else:
        target_volumes = all_volumes
        print(f"Downloading the full RadGenome-ChestCT '{split}' split ({len(all_volumes)} cases).")

    image_patterns = [volname_to_relpath[v] for v in target_volumes]
    if split == "train":
        mask_patterns = [f"dataset/train_anatomy_mask_a{c}" for c in "abcde"]
        print(
            "Downloading the training split's segmentation masks requires all 5 mask archive shards "
            "(~74 GB combined), regardless of how many cases are requested."
        )
    else:
        mask_patterns = ["dataset/valid_anatomy_mask.tar.gz"]
        print("Downloading the validation split's segmentation mask archive (~10.5 GB).")

    print(f"Downloading {len(target_volumes)} preprocessed CT volume(s) for split '{split}'.")
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", local_dir=path, allow_patterns=image_patterns + mask_patterns
    )

    if split == "train":
        tar_path = _concatenate_train_mask_shards(path)
    else:
        tar_path = os.path.join(path, "dataset", "valid_anatomy_mask.tar.gz")

    import tarfile

    mask_extract_dir = os.path.join(path, "dataset")
    wanted = [f"seg_{v}/" for v in target_volumes]
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if any(w in m.name for w in wanted)]
        tf.extractall(path=mask_extract_dir, members=members)

    volumes = [v for v in target_volumes if _find_mask_dir(path, v) is not None]
    if not volumes:
        raise RuntimeError(f"Could not find any RadGenome-ChestCT masks for the requested cases in '{path}'.")

    return volumes


def get_radgenome_chestct_paths(
    path: Union[os.PathLike, str],
    split: str = "validation",
    max_cases: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the RadGenome-ChestCT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split, either 'train' or 'validation'.
        max_cases: The maximum number of cases to use. See `get_radgenome_chestct_data` for details.
        case_ids: Explicit list of case ids to use. See `get_radgenome_chestct_data` for details.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    volumes = get_radgenome_chestct_data(path, split, max_cases, case_ids, download)

    raw_paths, label_paths = [], []
    for volume in tqdm(volumes, desc="Preparing RadGenome-ChestCT labels"):
        image_path = _find_image_path(path, volume)
        mask_dir = _find_mask_dir(path, volume)
        if image_path is None or mask_dir is None:
            continue
        raw_paths.append(image_path)
        label_paths.append(merge_segmentations(mask_dir))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    return raw_paths, label_paths


def get_radgenome_chestct_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: str = "validation",
    max_cases: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the RadGenome-ChestCT dataset for chest CT organ-level segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The data split, either 'train' or 'validation'.
        max_cases: The maximum number of cases to use. See `get_radgenome_chestct_data` for details.
        case_ids: Explicit list of case ids to use. See `get_radgenome_chestct_data` for details.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_radgenome_chestct_paths(path, split, max_cases, case_ids, download)

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


def get_radgenome_chestct_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: str = "validation",
    max_cases: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the RadGenome-ChestCT dataloader for chest CT organ-level segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split, either 'train' or 'validation'.
        max_cases: The maximum number of cases to use. See `get_radgenome_chestct_data` for details.
        case_ids: Explicit list of case ids to use. See `get_radgenome_chestct_data` for details.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch
            DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_radgenome_chestct_dataset(
        path, patch_shape, split, max_cases, case_ids, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
