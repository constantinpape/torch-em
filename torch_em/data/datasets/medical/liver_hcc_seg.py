"""The LiverHccSeg dataset contains annotations for liver and hepatocellular carcinoma (HCC)
tumor segmentation in multiphasic contrast-enhanced MRI.

The dataset is built from the multi-parametric MRI arm of TCGA-LIHC. It ships 4 native
acquisition phases ('pre', 'art', 'pv', 'del' - pre-contrast, arterial, portal-venous and
delayed) plus 3 phases registered onto the arterial phase ('art_pre', 'art_pv', 'art_del')
for each of 18 exams. Whole-liver masks are available for 17 exams and HCC tumor masks
(up to 3 lesions per exam) for 14 exams, each independently annotated by two board-certified
abdominal radiologists ('rater1' and 'rater2'). This module merges the per-lesion tumor masks
of an exam into a single instance label volume.

NOTE: One exam ships a raw volume and liver mask with a mismatched slice count; this loader skips
that pair, so 16 (not 17) liver exams are usable.

NOTE: This is MRI data and is not the same as the already-integrated
`torch_em.data.datasets.medical.waw_tace` or `torch_em.data.datasets.medical.hcc_tace`, which
are contrast-enhanced CT datasets, nor `torch_em.data.datasets.medical.openswisshcc`, which is a
different, larger multiphasic liver/HCC MRI cohort.

The data is located at https://doi.org/10.5281/zenodo.7957516, released under a CC-BY-4.0 license.

This dataset is from the publication https://doi.org/10.1016/j.dib.2023.109607.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/7957516/files/nifti_and_segms.zip"
CHECKSUM = "dba5f89a95b9c0cc4fec466fa8105663b754e7039ebee996b708ecf3ee114d7d"

PHASES = ("pre", "art", "pv", "del", "art_pre", "art_pv", "art_del")
RATERS = (1, 2)


def get_liver_hcc_seg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the LiverHccSeg dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "nifti_and_segms")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "nifti_and_segms.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _exam_dirs(data_dir):
    return natsorted(glob(os.path.join(data_dir, "TCGA-*", "*")))


def _merged_tumor_mask_path(exam_dir, rater):
    return os.path.join(exam_dir, f"rater{rater}_tumor_merged.nii.gz")


def _merge_tumor_masks(exam_dir, rater):
    """Merge the per-lesion tumor masks of one rater into a single instance label volume."""
    out_path = _merged_tumor_mask_path(exam_dir, rater)
    if os.path.exists(out_path):
        return out_path

    lesion_paths = natsorted(glob(os.path.join(exam_dir, f"rater{rater}_tumor*.nii.gz")))
    if not lesion_paths:
        return None

    import nibabel as nib

    merged = None
    reference = None
    for lesion_id, lesion_path in enumerate(lesion_paths, start=1):
        lesion_img = nib.load(lesion_path)
        if merged is None:
            reference = lesion_img
            merged = np.zeros(lesion_img.shape, dtype="uint8")
        merged[lesion_img.get_fdata() > 0] = lesion_id

    nib.save(nib.Nifti1Image(merged, reference.affine, reference.header), out_path)
    return out_path


def get_liver_hcc_seg_paths(
    path: Union[os.PathLike, str],
    phase: str = "pre",
    target: Literal["liver", "tumor"] = "liver",
    rater: Literal[1, 2] = 1,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the LiverHccSeg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        phase: The choice of MRI phase. One of 'pre', 'art', 'pv', 'del', 'art_pre', 'art_pv', 'art_del'.
        target: The choice of segmentation target. Either 'liver' (whole-liver mask) or
            'tumor' (merged instance mask of the annotated HCC lesions).
        rater: The choice of annotator. Either 1 or 2.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if phase not in PHASES:
        raise ValueError(f"'{phase}' is not a valid phase. Choose one of {PHASES}.")
    if target not in ("liver", "tumor"):
        raise ValueError(f"'{target}' is not a valid target. Choose 'liver' or 'tumor'.")
    if rater not in RATERS:
        raise ValueError(f"'{rater}' is not a valid rater. Choose one of {RATERS}.")

    import nibabel as nib

    data_dir = get_liver_hcc_seg_data(path, download)

    raw_paths, label_paths = [], []
    for exam_dir in _exam_dirs(data_dir):
        raw_path = os.path.join(exam_dir, f"{phase}.nii.gz")
        if not os.path.exists(raw_path):
            continue

        if target == "liver":
            label_path = os.path.join(exam_dir, f"rater{rater}_liver.nii.gz")
            if not os.path.exists(label_path):
                continue
        else:
            label_path = _merge_tumor_masks(exam_dir, rater)
            if label_path is None:
                continue

        # One exam (TCGA-DD-A4NH) ships a raw volume and liver mask with a mismatched slice count.
        # Skip such pairs rather than failing the whole dataset.
        if nib.load(raw_path).shape != nib.load(label_path).shape:
            continue

        raw_paths.append(raw_path)
        label_paths.append(label_path)

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    return raw_paths, label_paths


def get_liver_hcc_seg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    phase: str = "pre",
    target: Literal["liver", "tumor"] = "liver",
    rater: Literal[1, 2] = 1,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LiverHccSeg dataset for liver and HCC tumor segmentation in multiphasic MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        phase: The choice of MRI phase. One of 'pre', 'art', 'pv', 'del', 'art_pre', 'art_pv', 'art_del'.
        target: The choice of segmentation target. Either 'liver' or 'tumor'.
        rater: The choice of annotator. Either 1 or 2.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_liver_hcc_seg_paths(path, phase, target, rater, download)

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


def get_liver_hcc_seg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    phase: str = "pre",
    target: Literal["liver", "tumor"] = "liver",
    rater: Literal[1, 2] = 1,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LiverHccSeg dataloader for liver and HCC tumor segmentation in multiphasic MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        phase: The choice of MRI phase. One of 'pre', 'art', 'pv', 'del', 'art_pre', 'art_pv', 'art_del'.
        target: The choice of segmentation target. Either 'liver' or 'tumor'.
        rater: The choice of annotator. Either 1 or 2.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_liver_hcc_seg_dataset(path, patch_shape, phase, target, rater, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
