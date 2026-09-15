"""The AbdomenAtlas 1.1 Mini dataset contains annotations for 25 anatomical structures in abdominal CT scans.

The dataset consists of 9262 cases with per-structure binary masks and a combined semantic label volume.
Only the 5195 cases BDMAP_00000001 to BDMAP_00005195 contain the CT scan; for the remaining cases the CT scans have
to be obtained from the RSNA 2023 Abdominal Trauma Detection challenge (see the dataset page), so these cases are
skipped here. The label ids of the combined label volume ('combined_labels.nii.gz') are given in `CLASS_IDS`:
1: aorta, 2: gall bladder, 3: kidney (left), 4: kidney (right), 5: liver, 6: pancreas, 7: postcava, 8: spleen,
9: stomach, 10: adrenal gland (left), 11: adrenal gland (right), 12: bladder, 13: celiac trunk, 14: colon,
15: duodenum, 16: esophagus, 17: femur (left), 18: femur (right), 19: hepatic vessel, 20: intestine, 21: lung (left),
22: lung (right), 23: portal vein and splenic vein, 24: prostate, 25: rectum.
If the combined label volume is missing for a case, it is created by merging the per-structure masks in
'segmentations/<structure>.nii.gz' in the order of `CLASS_NAMES` (a structure with a higher id takes precedence).

The dataset is located at https://huggingface.co/datasets/AbdomenAtlas/_AbdomenAtlas1.1Mini. It is gated:
to download it, create a HuggingFace account, accept the terms and conditions on the dataset page and create an
access token (https://huggingface.co/settings/tokens). Pass the token via the `token` argument or the `HF_TOKEN`
environment variable. Alternatively, download and extract the dataset manually (see the dataset page) and pass the
folder that contains the 'BDMAP_XXXXXXXX' case folders (or their parent folder) as `path`.
The dataset is licensed under CC BY-NC-SA 4.0.

This dataset is from the publication https://doi.org/10.1016/j.media.2024.103285.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Optional

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


REPO_ID = "AbdomenAtlas/_AbdomenAtlas1.1Mini"

CLASS_NAMES = [
    "aorta", "gall_bladder", "kidney_left", "kidney_right", "liver", "pancreas", "postcava", "spleen", "stomach",
    "adrenal_gland_left", "adrenal_gland_right", "bladder", "celiac_trunk", "colon", "duodenum", "esophagus",
    "femur_left", "femur_right", "hepatic_vessel", "intestine", "lung_left", "lung_right",
    "portal_vein_and_splenic_vein", "prostate", "rectum",
]
"""The anatomical structures of the AbdomenAtlas 1.1 dataset. The label id of a structure is its 1-based index."""

CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
"""Mapping from the name of an anatomical structure to its label id in the combined label volumes."""


def merge_segmentations(case_dir: str) -> str:
    """Merge the per-structure binary masks of one AbdomenAtlas case into a single semantic label volume.

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


def _find_case_dirs(path):
    # NOTE: The archives do not all extract to the same depth. Most of them place the 'BDMAP_XXXXXXXX' folders
    # directly in the extraction folder, while the last two keep them inside a folder named after the archive.
    # So all depths have to be searched, otherwise the cases of the nested archives are silently missed.
    case_dirs = {}
    for pattern in ["BDMAP_*", os.path.join("*", "BDMAP_*"), os.path.join("*", "*", "BDMAP_*")]:
        for case_dir in glob(os.path.join(path, pattern)):
            if not os.path.isdir(case_dir):
                continue

            # If a case is found at multiple depths, then the folder that contains the image is preferred.
            case_name = os.path.basename(case_dir)
            if case_name not in case_dirs or os.path.exists(os.path.join(case_dir, "ct.nii.gz")):
                case_dirs[case_name] = case_dir

    # The folders are sorted by the case name, so that the order does not depend on the extraction depth.
    return [case_dirs[case_name] for case_name in natsorted(case_dirs)]


def get_abdomen_atlas_data(
    path: Union[os.PathLike, str], token: Optional[str] = None, download: bool = False
) -> List[str]:
    """Download the AbdomenAtlas 1.1 Mini dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        token: The HuggingFace access token. By default, the 'HF_TOKEN' environment variable is used.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the case folders.
    """
    case_dirs = _find_case_dirs(path)
    if case_dirs:
        return case_dirs

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False")

    token = os.environ.get("HF_TOKEN") if token is None else token
    if token is None:
        raise RuntimeError(
            "The AbdomenAtlas 1.1 Mini dataset is gated on HuggingFace. To download it: create a HuggingFace account, "
            f"accept the terms and conditions at https://huggingface.co/datasets/{REPO_ID}, create an access token at "
            "https://huggingface.co/settings/tokens and pass it via the 'token' argument or the 'HF_TOKEN' environment "
            "variable."
        )

    from huggingface_hub import snapshot_download

    os.makedirs(path, exist_ok=True)
    print("The AbdomenAtlas 1.1 Mini data is not available yet and will be downloaded.")
    print("Note that this dataset is very large (~300 GB), so this step can take several hours.")
    try:
        snapshot_download(
            repo_id=REPO_ID, repo_type="dataset", token=token, local_dir=path, allow_patterns=["*.tar.gz", "*.csv"]
        )
    except Exception as e:
        raise RuntimeError(
            f"The download of the AbdomenAtlas 1.1 Mini dataset failed ({e}). Please make sure that you have accepted "
            f"the terms and conditions at https://huggingface.co/datasets/{REPO_ID} with the account of the token."
        )

    for tar_path in natsorted(glob(os.path.join(path, "*.tar.gz"))):
        util.unzip_tarfile(tar_path=tar_path, dst=os.path.join(path, "uncompressed"), remove=False)

    case_dirs = _find_case_dirs(path)
    if not case_dirs:
        raise RuntimeError(f"Could not find the 'BDMAP_XXXXXXXX' case folders of the AbdomenAtlas dataset in '{path}'.")
    return case_dirs


def get_abdomen_atlas_paths(
    path: Union[os.PathLike, str], token: Optional[str] = None, download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the AbdomenAtlas 1.1 Mini data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        token: The HuggingFace access token. By default, the 'HF_TOKEN' environment variable is used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    case_dirs = get_abdomen_atlas_data(path, token, download)

    raw_paths, label_paths = [], []
    for case_dir in tqdm(case_dirs, desc="Preparing AbdomenAtlas labels"):
        raw_path = os.path.join(case_dir, "ct.nii.gz")
        if not os.path.exists(raw_path):  # The cases without CT are skipped.
            continue
        raw_paths.append(raw_path)
        label_paths.append(merge_segmentations(case_dir))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    return raw_paths, label_paths


def get_abdomen_atlas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    token: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AbdomenAtlas 1.1 Mini dataset for abdominal organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        token: The HuggingFace access token. By default, the 'HF_TOKEN' environment variable is used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_abdomen_atlas_paths(path, token, download)

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


def get_abdomen_atlas_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    token: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the AbdomenAtlas 1.1 Mini dataloader for abdominal organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        token: The HuggingFace access token. By default, the 'HF_TOKEN' environment variable is used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_abdomen_atlas_dataset(path, patch_shape, token, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
