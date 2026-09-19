"""The TotalSegmentator dataset contains annotations for 117 anatomical structures in CT scans.

Two versions of the dataset can be selected via the 'version' argument:
- 'v2' (default, for backward compatibility): 1228 CT volumes, located at https://doi.org/10.5281/zenodo.10047292.
- 'v3': 1939 CT volumes (291 additional pediatric CTs on top of the 1228 from 'v2', plus corrected and refined
  labels, especially for bones and vertebrae), located at https://doi.org/10.5281/zenodo.22688904.

Both versions ship an official split in 'meta.csv' ('v2': train / val / test, 'v3': train / test only, i.e. no
'val' split is available for 'v3') and provide each anatomical structure as a separate binary mask.
`get_totalsegmentator_data` merges these masks into a single semantic label volume per case, where the label id
of each structure is its (1-based) position in `CLASS_NAMES` for 'v2', or `CLASS_NAMES_V3` for 'v3' (see
`CLASS_IDS` for the 'v2' name -> id mapping). This is the class order of the 'total' task in the
TotalSegmentator repository (https://github.com/wasserth/TotalSegmentator). 'v3' relabels 'vertebrae_S1' as
'vertebrae_L6' as part of its corrected vertebrae labeling (verified on the released 'v3' archive), all other
116 structures are unchanged between 'v2' and 'v3'. The masks of a few structures may overlap, in this case the
structure with the higher label id takes precedence.

This dataset is from the publication https://doi.org/10.1148/ryai.230024.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from concurrent import futures
from typing import Union, Tuple, Literal, List, Optional

import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "v2": "https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip",
    "v3": "https://zenodo.org/records/22688904/files/Totalsegmentator_dataset_v300.zip",
}
CHECKSUMS = {
    "v2": "741dbc911a768e2ac2671c66d55332f7302ad624c915a57d08b142d8bdf0ca26",
    "v3": "b56ae18553853ff256fb0eef3a02e322fbc6c862d9015238c8c46b1b76fa027b",
}
DATA_DIRNAMES = {
    "v2": "Totalsegmentator_dataset_v201",
    "v3": "Totalsegmentator_dataset_v300",
}
# The 'v2' archive has no top-level folder (files extract directly into the data folder), while the
# 'v3' archive already contains a top-level folder matching 'DATA_DIRNAMES["v3"]'.
HAS_TOP_LEVEL_DIR = {"v2": False, "v3": True}

CLASS_NAMES = [
    "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas", "adrenal_gland_right",
    "adrenal_gland_left", "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right", "lung_lower_lobe_right", "esophagus", "trachea", "thyroid_gland", "small_bowel",
    "duodenum", "colon", "urinary_bladder", "prostate", "kidney_cyst_left", "kidney_cyst_right", "sacrum",
    "vertebrae_S1", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1", "vertebrae_T12",
    "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8", "vertebrae_T7", "vertebrae_T6", "vertebrae_T5",
    "vertebrae_T4", "vertebrae_T3", "vertebrae_T2", "vertebrae_T1", "vertebrae_C7", "vertebrae_C6", "vertebrae_C5",
    "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1", "heart", "aorta", "pulmonary_vein",
    "brachiocephalic_trunk", "subclavian_artery_right", "subclavian_artery_left", "common_carotid_artery_right",
    "common_carotid_artery_left", "brachiocephalic_vein_left", "brachiocephalic_vein_right", "atrial_appendage_left",
    "superior_vena_cava", "inferior_vena_cava", "portal_vein_and_splenic_vein", "iliac_artery_left",
    "iliac_artery_right", "iliac_vena_left", "iliac_vena_right", "humerus_left", "humerus_right", "scapula_left",
    "scapula_right", "clavicula_left", "clavicula_right", "femur_left", "femur_right", "hip_left", "hip_right",
    "spinal_cord", "gluteus_maximus_left", "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right",
    "gluteus_minimus_left", "gluteus_minimus_right", "autochthon_left", "autochthon_right", "iliopsoas_left",
    "iliopsoas_right", "brain", "skull", "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5",
    "rib_left_6", "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12", "rib_right_1",
    "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5", "rib_right_6", "rib_right_7", "rib_right_8",
    "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12", "sternum", "costal_cartilages",
]
"""The anatomical structures of the TotalSegmentator CT dataset. The label id of a structure is its 1-based index."""

CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
"""Mapping from the name of an anatomical structure to its label id in the merged label volumes."""

CLASS_NAMES_V3 = [name if name != "vertebrae_S1" else "vertebrae_L6" for name in CLASS_NAMES]
"""The anatomical structures of the 'v3' TotalSegmentator CT dataset. Identical to `CLASS_NAMES`, except that
'vertebrae_S1' was replaced by 'vertebrae_L6' as part of the corrected vertebrae labeling in 'v3' (verified on
the per-case 'segmentations' folders of the released 'v3' archive, all 117 structures are otherwise unchanged).
"""

CLASS_NAMES_BY_VERSION = {"v2": CLASS_NAMES, "v3": CLASS_NAMES_V3}


def merge_segmentations(case_dir: str, class_names: List[str], label_name: str = "labels.nii.gz") -> str:
    """Merge the per-class binary masks of one TotalSegmentator case into a single semantic label volume.

    The merged volume is stored as nifti next to the image. If it already exists it is not recomputed,
    so that a partially finished conversion can be resumed.

    Args:
        case_dir: The folder of the case, which contains the 'segmentations' sub-folder.
        class_names: The class names in label id order (the first class gets id 1).
        label_name: The filename of the merged label volume.

    Returns:
        The filepath to the merged label volume.
    """
    import nibabel as nib

    label_path = os.path.join(case_dir, label_name)
    if os.path.exists(label_path):
        return label_path

    labels, affine, header = None, None, None
    for class_id, class_name in enumerate(class_names, start=1):
        mask_nii = nib.load(os.path.join(case_dir, "segmentations", f"{class_name}.nii.gz"))
        mask = np.asarray(mask_nii.dataobj) > 0
        if labels is None:
            labels = np.zeros(mask.shape, dtype="uint8")
            affine, header = mask_nii.affine, mask_nii.header
        labels[mask] = class_id

    # Write to a temporary path first, so that an interrupted conversion is not mistaken for a complete one.
    tmp_path = os.path.join(case_dir, f"{label_name}.incomplete.nii.gz")
    nib.save(nib.Nifti1Image(labels, affine, header), tmp_path)
    os.replace(tmp_path, label_path)
    return label_path


def merge_all_segmentations(case_dirs: List[str], class_names: List[str], n_workers: Optional[int] = None) -> None:
    """Merge the per-class binary masks of all cases into semantic label volumes.

    Args:
        case_dirs: The case folders to process.
        class_names: The class names in label id order.
        n_workers: The number of parallel workers. By default the number of CPUs (at most 16) is used.
    """
    if all(os.path.exists(os.path.join(case_dir, "labels.nii.gz")) for case_dir in case_dirs):
        return

    if n_workers is None:
        n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        n_workers = min(16, n_cpus)

    with futures.ProcessPoolExecutor(n_workers) as pool:
        tasks = [pool.submit(merge_segmentations, case_dir, class_names) for case_dir in case_dirs]
        for task in tqdm(futures.as_completed(tasks), total=len(tasks), desc="Merge the per-class segmentations"):
            task.result()


def read_split(meta_csv: str, split: str, valid_splits: Tuple[str, ...] = ("train", "val", "test")) -> List[str]:
    """Read the case ids of a split from the TotalSegmentator 'meta.csv'.

    Args:
        meta_csv: The path to the 'meta.csv' file.
        split: The choice of data split.
        valid_splits: The splits available in this dataset.

    Returns:
        The case ids of the split.
    """
    import pandas as pd

    if split not in valid_splits:
        raise ValueError(f"'{split}' is not a valid split. Choose one of {valid_splits}.")

    meta = pd.read_csv(meta_csv, sep=";", encoding="utf-8-sig")
    return sorted(meta[meta["split"] == split]["image_id"].tolist())


def get_totalsegmentator_data(
    path: Union[os.PathLike, str],
    download: bool = False,
    n_workers: Optional[int] = None,
    version: Literal["v2", "v3"] = "v2",
) -> str:
    """Download the TotalSegmentator CT dataset and merge the per-class masks into semantic label volumes.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.
        n_workers: The number of parallel workers for merging the per-class masks.
        version: The version of the dataset. Either 'v2' (1228 CTs) or 'v3' (1939 CTs, incl. pediatric CTs
            and refined bone / vertebrae labels).

    Returns:
        Filepath where the data is downloaded.
    """
    if version not in URLS:
        raise ValueError(f"'{version}' is not a valid version. Choose one of {list(URLS.keys())}.")

    dirname = DATA_DIRNAMES[version]
    data_dir = os.path.join(path, dirname)
    if not os.path.exists(os.path.join(data_dir, "meta.csv")):
        os.makedirs(path, exist_ok=True)
        zip_path = os.path.join(path, f"{dirname}.zip")
        util.download_source(path=zip_path, url=URLS[version], download=download, checksum=CHECKSUMS[version])
        # Extract into 'data_dir' if the archive has no top-level folder, otherwise into 'path'
        # (the archive's own top-level folder then becomes 'data_dir').
        util.unzip(zip_path=zip_path, dst=path if HAS_TOP_LEVEL_DIR[version] else data_dir)

    case_dirs = sorted(glob(os.path.join(data_dir, "s*")))
    merge_all_segmentations(case_dirs, CLASS_NAMES_BY_VERSION[version], n_workers)

    return data_dir


def get_totalsegmentator_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'val', 'test'],
    download: bool = False,
    version: Literal["v2", "v3"] = "v2",
) -> Tuple[List[str], List[str]]:
    """Get paths to the TotalSegmentator CT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.
        version: The version of the dataset. Either 'v2' (1228 CTs) or 'v3' (1939 CTs).

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_totalsegmentator_data(path, download, version=version)
    case_ids = read_split(os.path.join(data_dir, "meta.csv"), split)

    raw_paths = [os.path.join(data_dir, case_id, "ct.nii.gz") for case_id in case_ids]
    label_paths = [os.path.join(data_dir, case_id, "labels.nii.gz") for case_id in case_ids]
    assert all(os.path.exists(p) for p in raw_paths + label_paths)

    return raw_paths, label_paths


def get_totalsegmentator_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    resize_inputs: bool = False,
    download: bool = False,
    version: Literal["v2", "v3"] = "v2",
    **kwargs
) -> Dataset:
    """Get the TotalSegmentator dataset for segmentation of anatomical structures in CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        version: The version of the dataset. Either 'v2' (1228 CTs, default) or 'v3' (1939 CTs).
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_totalsegmentator_paths(path, split, download, version=version)

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


def get_totalsegmentator_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    resize_inputs: bool = False,
    download: bool = False,
    version: Literal["v2", "v3"] = "v2",
    **kwargs
) -> DataLoader:
    """Get the TotalSegmentator dataloader for segmentation of anatomical structures in CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        version: The version of the dataset. Either 'v2' (1228 CTs, default) or 'v3' (1939 CTs).
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_totalsegmentator_dataset(path, patch_shape, split, resize_inputs, download, version, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
