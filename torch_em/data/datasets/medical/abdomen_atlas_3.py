"""The AbdomenAtlas 3.0 Mini dataset contains annotations for 44 anatomical structures
(organs, blood vessels, liver/pancreas sub-segments and tumor lesions) in abdominal CT scans.

The dataset consists of 9262 cases (BDMAP_00000001 to BDMAP_00009262), each with a CT scan
('ct.nii.gz') and per-structure binary masks in a 'segmentations' sub-folder. Unlike AbdomenAtlas 1.1 Mini
(see `abdomen_atlas.py`), it is a public, ungated release: no HuggingFace account or access token is needed.
The data is hosted as 40 shards of (mostly) 232 cases each, separately for the CT scans ('image_only/*.tar.gz',
~14 GB per shard) and the segmentation masks ('mask_only/*.tar.gz', ~320 MB per shard), for a combined size of
about 586 GB. Downloading a case therefore always downloads the whole shard it belongs to; use `max_cases` or
`case_ids` to only fetch the shards required for a small subset of cases (see below).

The label ids of the combined semantic label volume created by `merge_segmentations` are given in `CLASS_IDS`:
1: adrenal_gland_left, 2: adrenal_gland_right, 3: bladder, 4: colon, 5: duodenum, 6: esophagus, 7: femur_left,
8: femur_right, 9: gall_bladder, 10: intestine, 11: kidney_left, 12: kidney_right, 13: liver, 14: lung_left,
15: lung_right, 16: pancreas, 17: prostate, 18: rectum, 19: spleen, 20: stomach, 21: aorta, 22: celiac_aa,
23: celiac_trunk, 24: common_bile_duct, 25: hepatic_vessel, 26: portal_vein_and_splenic_vein, 27: postcava,
28: superior_mesenteric_artery, 29: veins, 30: liver_segment_1, 31: liver_segment_2, 32: liver_segment_3,
33: liver_segment_4, 34: liver_segment_5, 35: liver_segment_6, 36: liver_segment_7, 37: liver_segment_8,
38: pancreas_head, 39: pancreas_body, 40: pancreas_tail, 41: liver_lesion, 42: kidney_lesion,
43: pancreatic_lesion, 44: colon_lesion.
The tumor lesion classes (41-44) are merged last, so they take precedence over the organ they lie in wherever
the lesion and organ masks overlap. 'colon_lesion' is only present for a small subset of cases (the mask files
for the other 43 structures are provided for every case, even if empty).
If the combined label volume is missing for a case, it is created by merging the per-structure masks in
'segmentations/<structure>.nii.gz' in the order of `CLASS_NAMES` (a structure with a higher id takes precedence).

The dataset is located at https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini. It is licensed
under CC BY-NC-SA 4.0.

This dataset is from the publication https://arxiv.org/abs/2501.04678.
Please cite it if you use this dataset in your research.
"""

import os
import re
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Optional

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util
from .abdomen_atlas import _find_case_dirs


REPO_ID = "AbdomenAtlas/AbdomenAtlas3.0Mini"

ORGAN_NAMES = [
    "adrenal_gland_left", "adrenal_gland_right", "bladder", "colon", "duodenum", "esophagus", "femur_left",
    "femur_right", "gall_bladder", "intestine", "kidney_left", "kidney_right", "liver", "lung_left", "lung_right",
    "pancreas", "prostate", "rectum", "spleen", "stomach",
]
"""The organs of the AbdomenAtlas 3.0 dataset."""

VESSEL_NAMES = [
    "aorta", "celiac_aa", "celiac_trunk", "common_bile_duct", "hepatic_vessel", "portal_vein_and_splenic_vein",
    "postcava", "superior_mesenteric_artery", "veins",
]
"""The blood vessels and bile duct of the AbdomenAtlas 3.0 dataset."""

SUBSEGMENT_NAMES = [
    "liver_segment_1", "liver_segment_2", "liver_segment_3", "liver_segment_4", "liver_segment_5",
    "liver_segment_6", "liver_segment_7", "liver_segment_8", "pancreas_head", "pancreas_body", "pancreas_tail",
]
"""The liver and pancreas sub-segments of the AbdomenAtlas 3.0 dataset."""

LESION_NAMES = ["liver_lesion", "kidney_lesion", "pancreatic_lesion", "colon_lesion"]
"""The tumor lesion classes of the AbdomenAtlas 3.0 dataset. These are merged last, ie. they take precedence
over the organ they lie in wherever the lesion and organ masks overlap."""

CLASS_NAMES = ORGAN_NAMES + VESSEL_NAMES + SUBSEGMENT_NAMES + LESION_NAMES
"""All structures of the AbdomenAtlas 3.0 dataset, in the order used to build the combined label volume."""

CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
"""Mapping from the name of a structure to its label id in the combined label volumes."""


def merge_segmentations(case_dir: str) -> str:
    """Merge the per-structure binary masks of one AbdomenAtlas 3.0 case into a single semantic label volume.

    The merged volume is stored as 'combined_labels.nii.gz' in the case folder. If it already exists,
    it is not recomputed.

    Args:
        case_dir: The folder of the case, which contains the 'segmentations' sub-folder.

    Returns:
        The filepath to the merged label volume.
    """
    import nibabel as nib

    label_path = os.path.join(case_dir, "combined_labels.nii.gz")
    if os.path.exists(label_path):
        return label_path

    labels, affine = None, None
    for class_name in CLASS_NAMES:
        mask_path = os.path.join(case_dir, "segmentations", f"{class_name}.nii.gz")
        if not os.path.exists(mask_path):
            continue
        nifti = nib.load(mask_path)
        mask = np.asarray(nifti.dataobj) > 0
        if labels is None:
            labels, affine = np.zeros(mask.shape, dtype="uint8"), nifti.affine
        labels[mask] = CLASS_IDS[class_name]

    if labels is None:
        raise RuntimeError(f"Could not find any segmentation masks in '{case_dir}'.")

    nib.save(nib.Nifti1Image(labels, affine), label_path)
    return label_path


def _parse_shards(files, subdir):
    # Parse the ('AbdomenAtlas3_images_BDMAP_BDMAP_00000001_BDMAP_00000232.tar.gz'-style) shard filenames
    # into (start_case, end_case, filename) tuples, so that the shard(s) covering a given case can be found.
    shards = []
    for fpath in files:
        if not (fpath.startswith(f"{subdir}/") and fpath.endswith(".tar.gz")):
            continue
        numbers = re.findall(r"BDMAP_(\d{8})", os.path.basename(fpath))
        assert len(numbers) == 2, f"Could not parse the case range from '{fpath}'."
        shards.append((int(numbers[0]), int(numbers[1]), fpath))
    assert shards, f"Could not find any shards under '{subdir}' in the AbdomenAtlas 3.0 repository."
    return sorted(shards)


def _shards_for_cases(shards, case_numbers):
    patterns = set()
    for start, end, fpath in shards:
        if any(start <= number <= end for number in case_numbers):
            patterns.add(fpath)
    return patterns


def get_abdomen_atlas_3_data(
    path: Union[os.PathLike, str],
    max_cases: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
    download: bool = False,
) -> List[str]:
    """Download the AbdomenAtlas 3.0 Mini dataset.

    The dataset is ungated and does not require a HuggingFace account or access token. It is, however, very
    large (~586 GB for all 9262 cases, split into 40 shards of ~14 GB (images) + ~320 MB (masks) each), and a
    case can only be downloaded together with the full shard (of up to 232 cases) it belongs to. Use `max_cases`
    or `case_ids` to only download the shards required for a small subset of cases; leave both at their default
    (None) to download the full dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        max_cases: The maximum number of cases to download, taken in order starting from 'BDMAP_00000001'.
            Only the shards covering these cases are downloaded. Mutually exclusive with `case_ids`.
        case_ids: Explicit list of case ids (eg. ['BDMAP_00000001', 'BDMAP_00000002']) to download. Only the
            shards covering these cases are downloaded. Mutually exclusive with `max_cases`.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the case folders.
    """
    assert max_cases is None or case_ids is None, "'max_cases' and 'case_ids' are mutually exclusive."

    case_dirs = _find_case_dirs(path)
    if case_dirs:
        if max_cases is not None:
            case_dirs = case_dirs[:max_cases]
        elif case_ids is not None:
            case_dirs = [c for c in case_dirs if os.path.basename(c) in case_ids]
        if case_dirs:
            return case_dirs

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False")

    from huggingface_hub import HfApi, snapshot_download

    os.makedirs(path, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    image_shards = _parse_shards(files, "image_only")
    mask_shards = _parse_shards(files, "mask_only")

    if case_ids is not None:
        case_numbers = [int(cid.split("_")[-1]) for cid in case_ids]
    elif max_cases is not None:
        last_case = image_shards[-1][1]
        case_numbers = list(range(1, min(max_cases, last_case) + 1))
    else:
        case_numbers = None  # Download everything.

    if case_numbers is None:
        allow_patterns = ["*.tar.gz", "*.csv"]
        print("The AbdomenAtlas 3.0 Mini data is not available yet and will be downloaded.")
        print("Note that this dataset is very large (~586 GB), so this step can take several hours.")
    else:
        allow_patterns = sorted(
            _shards_for_cases(image_shards, case_numbers) | _shards_for_cases(mask_shards, case_numbers)
        ) + ["*.csv"]
        print(f"Downloading the AbdomenAtlas 3.0 Mini shards required for {len(case_numbers)} case(s).")

    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=path, allow_patterns=allow_patterns)

    for tar_path in natsorted(glob(os.path.join(path, "*", "*.tar.gz"))):
        util.unzip_tarfile(tar_path=tar_path, dst=os.path.join(path, "uncompressed"), remove=False)

    case_dirs = _find_case_dirs(path)
    if not case_dirs:
        raise RuntimeError(
            f"Could not find the 'BDMAP_XXXXXXXX' case folders of the AbdomenAtlas 3.0 dataset in '{path}'."
        )

    if max_cases is not None:
        case_dirs = case_dirs[:max_cases]
    elif case_ids is not None:
        case_dirs = [c for c in case_dirs if os.path.basename(c) in case_ids]

    return case_dirs


def get_abdomen_atlas_3_paths(
    path: Union[os.PathLike, str],
    max_cases: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the AbdomenAtlas 3.0 Mini data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        max_cases: The maximum number of cases to use, taken in order starting from 'BDMAP_00000001'.
            See `get_abdomen_atlas_3_data` for details on how this restricts the download volume.
        case_ids: Explicit list of case ids to use. See `get_abdomen_atlas_3_data` for details.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    case_dirs = get_abdomen_atlas_3_data(path, max_cases, case_ids, download)

    raw_paths, label_paths = [], []
    for case_dir in tqdm(case_dirs, desc="Preparing AbdomenAtlas 3.0 labels"):
        raw_path = os.path.join(case_dir, "ct.nii.gz")
        if not os.path.exists(raw_path):
            continue
        raw_paths.append(raw_path)
        label_paths.append(merge_segmentations(case_dir))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    return raw_paths, label_paths


def get_abdomen_atlas_3_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    max_cases: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AbdomenAtlas 3.0 Mini dataset for abdominal organ, vessel and tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        max_cases: The maximum number of cases to use. See `get_abdomen_atlas_3_data` for details.
        case_ids: Explicit list of case ids to use. See `get_abdomen_atlas_3_data` for details.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_abdomen_atlas_3_paths(path, max_cases, case_ids, download)

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


def get_abdomen_atlas_3_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    max_cases: Optional[int] = None,
    case_ids: Optional[List[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the AbdomenAtlas 3.0 Mini dataloader for abdominal organ, vessel and tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        max_cases: The maximum number of cases to use. See `get_abdomen_atlas_3_data` for details.
        case_ids: Explicit list of case ids to use. See `get_abdomen_atlas_3_data` for details.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_abdomen_atlas_3_dataset(path, patch_shape, max_cases, case_ids, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
