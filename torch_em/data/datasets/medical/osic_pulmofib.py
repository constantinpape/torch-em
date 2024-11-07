"""The OSIC PulmoFib dataset contains annotations for lung, heart and trachea in CT scans.

This dataset is from OSIC Pulmonary Fibrosis Progression Challenge:
- https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/data (dataset source)
- https://www.kaggle.com/datasets/sandorkonya/ct-lung-heart-trachea-segmentation (segmentation source)
Please cite them if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List

import json
import numpy as np

import torch_em

from .. import util


ORGAN_IDS = {"heart": 1, "lung": 2, "trachea": 3}


def get_osic_pulmofib_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the OSIC PulmoFib dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    # download the inputs
    zip_path = os.path.join(path, "osic-pulmonary-fibrosis-progression.zip")
    util.download_source_kaggle(
        path=path, dataset_name="osic-pulmonary-fibrosis-progression", download=download, competition=True
    )
    util.unzip(zip_path=zip_path, dst=data_dir, remove=False)

    # download the labels
    zip_path = os.path.join(path, "ct-lung-heart-trachea-segmentation.zip")
    util.download_source_kaggle(
        path=path, dataset_name="sandorkonya/ct-lung-heart-trachea-segmentation", download=download
    )
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def get_osic_pulmofib_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the OSIC PulmoFib data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    import nrrd
    import nibabel as nib
    import pydicom as dicom

    data_dir = get_osic_pulmofib_data(path=path, download=download)

    image_dir = os.path.join(data_dir, "preprocessed", "images")
    gt_dir = os.path.join(data_dir, "preprocessed", "ground_truth")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    cpath = os.path.join(data_dir, "preprocessed", "confirmer.json")
    _completed_preproc = os.path.exists(cpath)

    image_paths, gt_paths = [], []
    uid_paths = natsorted(glob(os.path.join(data_dir, "train", "*")))
    for uid_path in tqdm(uid_paths, desc="Preprocessing inputs"):
        uid = uid_path.split("/")[-1]

        image_path = os.path.join(image_dir, f"{uid}.nii.gz")
        gt_path = os.path.join(gt_dir, f"{uid}.nii.gz")

        if _completed_preproc:
            if os.path.exists(image_path) and os.path.exists(gt_path):
                image_paths.append(image_path)
                gt_paths.append(gt_path)

            continue

        # creating the volume out of individual dicom slices
        all_slices = []
        for slice_path in natsorted(glob(os.path.join(uid_path, "*.dcm"))):
            per_slice = dicom.dcmread(slice_path)
            per_slice = per_slice.pixel_array
            all_slices.append(per_slice)
        all_slices = np.stack(all_slices).transpose(1, 2, 0)

        # next, combining the semantic organ annotations into one ground-truth volume with specific semantic labels
        all_gt = np.zeros(all_slices.shape, dtype="uint8")
        for ann_path in glob(os.path.join(data_dir, "*", "*", f"{uid}_*.nrrd")):
            ann_organ = Path(ann_path).stem.split("_")[-1]
            if ann_organ == "noisy":
                continue

            per_gt, _ = nrrd.read(ann_path)
            per_gt = per_gt.transpose(1, 0, 2)

            # some organ anns have weird dimension mismatch, we don't consider them for simplicity
            if per_gt.shape == all_slices.shape:
                all_gt[per_gt > 0] = ORGAN_IDS[ann_organ]

        # only if the volume has any labels (some volumes do not have segmentations), we save those raw and gt volumes
        if len(np.unique(all_gt)) > 1:
            all_gt = np.flip(all_gt, axis=2)

            image_nifti = nib.Nifti2Image(all_slices, np.eye(4))
            gt_nifti = nib.Nifti2Image(all_gt, np.eye(4))

            nib.save(image_nifti, image_path)
            nib.save(gt_nifti, gt_path)

            image_paths.append(image_path)
            gt_paths.append(gt_path)

    if not _completed_preproc:
        # since we do not have segmentation for all volumes, we store a file which reflects aggrement of created dataset
        confirm_msg = "The dataset has been preprocessed. "
        confirm_msg += f"It has {len(image_paths)} volume and {len(gt_paths)} respective ground-truth."
        print(confirm_msg)

        with open(cpath, "w") as f:
            json.dump(confirm_msg, f)

    return image_paths, gt_paths


def get_osic_pulmofib_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Get the OSIC PulmoFib dataset for segmentation of lung, heart and trachea.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_osic_pulmofib_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        **kwargs
    )


def get_osic_pulmofib_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Get the OSIC PulmoFib dataloader for segmentation of lung, heart and trachea.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_osic_pulmofib_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
