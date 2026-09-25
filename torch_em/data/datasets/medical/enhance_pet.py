"""The ENHANCE.PET 1.6k dataset contains annotations for 130 anatomical structures in whole-/total-body CT
volumes, co-acquired with [18F]FDG-PET as part of routine oncological imaging (~1,597 studies).

The per-structure binary masks are grouped into 7 category volumes ('Body-Composition', 'Cardiac', 'Muscles',
'Organs', 'Peripheral-Bones', 'Ribs', 'Vertebrae'), each registered to the CT grid, with structure ids
restarting at 1 within each category. `get_enhance_pet_data` merges the 7 category volumes of a case into a
single semantic label volume, where the label id of a structure is its (1-based) position in `CLASS_NAMES`
(see `CLASS_NAMES_BY_CATEGORY` for the per-category structure lists this is built from). The PET volumes have
a different grid than the CT / label volumes (no shared voxel grid without resampling) and are not exposed by
this loader, only the CT volumes are used as raw data.

The data is located at the public, anonymously readable S3 bucket 'enhance-pet-1-6k'
(https://registry.opendata.aws/enhance-pet-1-6k/, region 'us-west-2'), released under a CC-BY-4.0 license.

This dataset is from the publication https://doi.org/10.1038/s41597-026-07218-y.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from concurrent import futures
from typing import Union, Tuple, List, Optional

import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BUCKET_URL = "https://enhance-pet-1-6k.s3.us-west-2.amazonaws.com"

CLASS_NAMES_BY_CATEGORY = {
    "Body-Composition": ["skeletal_muscle", "subcutaneous_fat", "visceral_fat"],
    "Cardiac": [
        "heart_myocardium", "heart_atrium_left", "heart_atrium_right", "heart_ventricle_left",
        "heart_ventricle_right", "aorta", "iliac_artery_left", "iliac_artery_right", "iliac_vena_left",
        "iliac_vena_right", "inferior_vena_cava", "portal_splenic_vein", "pulmonary_artery",
    ],
    "Muscles": [
        "autochthon_left", "autochthon_right", "gluteus_maximus_left", "gluteus_maximus_right",
        "gluteus_medius_left", "gluteus_medius_right", "gluteus_minimus_left", "gluteus_minimus_right",
        "iliopsoas_left", "iliopsoas_right",
    ],
    "Organs": [
        "adrenal_gland_left", "adrenal_gland_right", "bladder", "brain", "gallbladder", "kidney_left",
        "kidney_right", "liver", "lung_lower_lobe_left", "lung_lower_lobe_right", "lung_middle_lobe_right",
        "lung_upper_lobe_left", "lung_upper_lobe_right", "pancreas", "spleen", "stomach", "thyroid_left",
        "thyroid_right",
    ],
    "Peripheral-Bones": [
        "carpal_left", "carpal_right", "clavicle_left", "clavicle_right", "femur_left", "femur_right",
        "fibula_left", "fibula_right", "fingers_left", "fingers_right", "humerus_left", "humerus_right",
        "metacarpal_left", "metacarpal_right", "metatarsal_left", "metatarsal_right", "patella_left",
        "patella_right", "radius_left", "radius_right", "scapula_left", "scapula_right", "skull", "tarsal_left",
        "tarsal_right", "tibia_left", "tibia_right", "toes_left", "toes_right", "ulna_left", "ulna_right",
    ],
    "Ribs": [f"rib_left_{i}" for i in range(1, 14)] + [f"rib_right_{i}" for i in range(1, 14)] + ["sternum"],
    "Vertebrae": (
        [f"vertebra_C{i}" for i in range(1, 8)] + [f"vertebra_T{i}" for i in range(1, 13)]
        + [f"vertebra_L{i}" for i in range(1, 7)] + ["hip_left", "hip_right", "sacrum"]
    ),
}
"""The anatomical structures of the ENHANCE.PET 1.6k dataset, grouped by their ground-truth category volume."""

CATEGORIES = sorted(CLASS_NAMES_BY_CATEGORY.keys())

CLASS_NAMES = [name for category in CATEGORIES for name in CLASS_NAMES_BY_CATEGORY[category]]
"""The anatomical structures of the ENHANCE.PET 1.6k dataset. The label id of a structure is its 1-based index."""

CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
"""Mapping from the name of an anatomical structure to its label id in the merged label volumes."""


def _list_case_ids(n_workers=None):
    import requests
    from xml.etree import ElementTree

    case_ids, token = [], None
    while True:
        params = {"list-type": "2", "prefix": "imaging-data/images/CT/"}
        if token:
            params["continuation-token"] = token
        response = requests.get(BUCKET_URL, params=params)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for content in root.findall("s3:Contents", ns):
            key = content.find("s3:Key", ns).text
            case_ids.append(os.path.basename(key).replace(".nii.gz", ""))
        is_truncated = root.find("s3:IsTruncated", ns).text == "true"
        if not is_truncated:
            break
        token = root.find("s3:NextContinuationToken", ns).text

    return sorted(case_ids)


def _merge_segmentations(path: str, case_id: str) -> str:
    """Merge the 7 per-category label volumes of one case into a single semantic label volume."""
    import nibabel as nib

    label_path = os.path.join(path, "imaging-data", "labels", f"{case_id}.nii.gz")
    if os.path.exists(label_path):
        return label_path

    labels, affine, header, offset = None, None, None, 0
    for category in CATEGORIES:
        category_path = os.path.join(path, "imaging-data", "ground-truth", category, f"{case_id}.nii.gz")
        category_nii = nib.load(category_path)
        category_labels = np.round(np.asarray(category_nii.dataobj)).astype("uint8")
        if labels is None:
            labels = np.zeros(category_labels.shape, dtype="uint8")
            affine, header = category_nii.affine, category_nii.header
        foreground = category_labels > 0
        labels[foreground] = category_labels[foreground] + offset
        offset += len(CLASS_NAMES_BY_CATEGORY[category])

    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    tmp_path = label_path.replace(".nii.gz", ".incomplete.nii.gz")
    nib.save(nib.Nifti1Image(labels, affine, header), tmp_path)
    os.replace(tmp_path, label_path)
    return label_path


def _download_case(case_id: str, path: str) -> None:
    image_path = os.path.join(path, "imaging-data", "images", "CT", f"{case_id}.nii.gz")
    util.download_source(
        path=image_path, url=f"{BUCKET_URL}/imaging-data/images/CT/{case_id}.nii.gz", download=True,
    )
    for category in CATEGORIES:
        category_path = os.path.join(path, "imaging-data", "ground-truth", category, f"{case_id}.nii.gz")
        util.download_source(
            path=category_path,
            url=f"{BUCKET_URL}/imaging-data/ground-truth/{category}/{case_id}.nii.gz", download=True,
        )
    _merge_segmentations(path, case_id)


def get_enhance_pet_data(
    path: Union[os.PathLike, str], n_cases: Optional[int] = None, n_workers: Optional[int] = None,
    download: bool = False,
) -> str:
    """Download the ENHANCE.PET 1.6k dataset and merge the per-category masks into semantic label volumes.

    NOTE: The full collection is about 250 GB. Use `n_cases` to only download a subset for a quick start.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        n_cases: The number of cases to download, sorted by case id. By default all ~1597 cases are downloaded.
        n_workers: The number of parallel download / merging workers. By default the number of CPUs (at most
            16) is used.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    os.makedirs(path, exist_ok=True)
    case_ids = _list_case_ids()
    if n_cases is not None:
        case_ids = case_ids[:n_cases]

    missing = [
        case_id for case_id in case_ids
        if not os.path.exists(os.path.join(path, "imaging-data", "labels", f"{case_id}.nii.gz"))
    ]
    if missing and not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    if n_workers is None:
        n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        n_workers = min(16, n_cpus)

    with futures.ThreadPoolExecutor(n_workers) as pool:
        tasks = [pool.submit(_download_case, case_id, path) for case_id in missing]
        for task in tqdm(futures.as_completed(tasks), total=len(tasks), desc="Download ENHANCE.PET cases"):
            task.result()

    return path


def get_enhance_pet_paths(
    path: Union[os.PathLike, str], n_cases: Optional[int] = None, download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the ENHANCE.PET 1.6k data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        n_cases: The number of cases to use, sorted by case id. By default all downloaded cases are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the CT image data.
        List of filepaths for the label data.
    """
    get_enhance_pet_data(path, n_cases, download=download)

    raw_paths = sorted(glob(os.path.join(path, "imaging-data", "images", "CT", "*.nii.gz")))
    label_paths = [
        os.path.join(path, "imaging-data", "labels", os.path.basename(p)) for p in raw_paths
    ]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in label_paths)
    return raw_paths, label_paths


def get_enhance_pet_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    n_cases: Optional[int] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ENHANCE.PET 1.6k dataset for segmentation of anatomical structures in whole-body CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        n_cases: The number of cases to use, sorted by case id. By default all downloaded cases are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_enhance_pet_paths(path, n_cases, download)

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
        is_seg_dataset=True,
        patch_shape=patch_shape,
        ndim=3,
        **kwargs
    )


def get_enhance_pet_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    n_cases: Optional[int] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ENHANCE.PET 1.6k dataloader for segmentation of anatomical structures in whole-body CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        n_cases: The number of cases to use, sorted by case id. By default all downloaded cases are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_enhance_pet_dataset(path, patch_shape, n_cases, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
