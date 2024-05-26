import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple

import nrrd
import numpy as np
import nibabel as nib
import pydicom as dicom

import torch_em

from .. import util


ORGAN_IDS = {
    "heart": 1, "lung": 2, "trachea": 3,
}


def get_osic_pulmofib_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    # download the data first
    zip_path = os.path.join(path, "osic-pulmonary-fibrosis-progression.zip")
    util.download_source_kaggle(
        path=path, dataset_name="", download=download
    )
    util.unzip(zip_path=zip_path, dst=data_dir, remove=False)

    # download the ground truth next
    zip_path = os.path.join(path, "ct-lung-heart-trachea-segmentation.zip")
    util.download_source_kaggle(
        path=path, dataset_name="sandorkonya/ct-lung-heart-trachea-segmentation", download=download
    )
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def _get_osic_pulmofib_paths(path, download):
    data_dir = get_osic_pulmofib_data(path=path, download=download)

    image_dir = os.path.join(data_dir, "preprocessed", "images")
    gt_dir = os.path.join(data_dir, "preprocessed", "ground_truth")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    image_paths, gt_paths = [], []
    uid_paths = natsorted(glob(os.path.join(data_dir, "train", "*")))
    for uid_path in tqdm(uid_paths):
        uid = uid_path.split("/")[-1]

        image_path = os.path.join(image_dir, f"{uid}.nii.gz")
        gt_path = os.path.join(gt_dir, f"{uid}.nii.gz")
        if os.path.exists(image_path) and os.path.exists(gt_path):
            continue

        all_slices = []
        for slice_path in natsorted(glob(os.path.join(uid_path, "*.dcm"))):
            per_slice = dicom.dcmread(slice_path)
            per_slice = per_slice.pixel_array
            all_slices.append(per_slice)

        all_slices = np.stack(all_slices).transpose(1, 2, 0)

        all_gt = np.zeros(all_slices.shape, dtype="uint8").transpose(1, 0, 2)
        for ann_path in glob(os.path.join(data_dir, "*", "*", f"{uid}_*.nrrd")):
            ann_organ = Path(ann_path).stem.split("_")[-1]
            if ann_organ == "noisy":
                continue

            per_gt, _ = nrrd.read(ann_path)

            try:
                all_gt[per_gt > 0] = ORGAN_IDS[ann_organ]
            except IndexError:  # some organ anns have weird shapes, for simplicity we don't consider them.
                if per_gt.shape != all_slices.shape:
                    continue

        if len(np.unique(all_gt)) > 1:
            all_gt = np.flip(all_gt, axis=2)

            # image_nifti = nib.Nifti2Image(all_slices, np.eye(4))
            # gt_nifti = nib.Nifti2Image(all_gt, np.eye(4))

            # nib.save(image_nifti, image_path)
            # nib.save(gt_nifti, gt_path)

        print(np.unique(all_gt))

    breakpoint()

    return image_paths, gt_paths


def get_osic_pulmofib_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """
    """
    image_paths, gt_paths = _get_osic_pulmofib_paths(path=path, download=download)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_osic_pulmofib_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_osic_pulmofib_dataset(
        path=path, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
