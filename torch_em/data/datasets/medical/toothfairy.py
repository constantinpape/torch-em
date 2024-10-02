import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted

import numpy as np

import torch_em

from .. import util


def get_toothfairy_data(path, download):
    """Automatic download is not possible.
    """
    if download:
        raise NotImplementedError

    data_dir = os.path.join(path, "ToothFairy_Dataset", "Dataset")
    return data_dir


def _get_toothfairy_paths(path, download):
    import nibabel as nib

    data_dir = get_toothfairy_data(path, download)

    images_dir = os.path.join(path, "data", "images")
    gt_dir = os.path.join(path, "data", "dense_labels")
    if os.path.exists(images_dir) and os.path.exists(gt_dir):
        return natsorted(glob(os.path.join(images_dir, "*.nii.gz"))), natsorted(glob(os.path.join(gt_dir, "*.nii.gz")))

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    image_paths, gt_paths = [], []
    for patient_dir in tqdm(glob(os.path.join(data_dir, "P*"))):
        patient_id = os.path.split(patient_dir)[-1]

        dense_anns_path = os.path.join(patient_dir, "gt_alpha.npy")
        if not os.path.exists(dense_anns_path):
            continue

        image_path = os.path.join(patient_dir, "data.npy")

        image = np.load(image_path)
        gt = np.load(dense_anns_path)

        image_nifti = nib.Nifti2Image(image, np.eye(4))
        gt_nifti = nib.Nifti2Image(gt, np.eye(4))

        trg_image_path = os.path.join(images_dir, f"{patient_id}.nii.gz")
        trg_gt_path = os.path.join(gt_dir, f"{patient_id}.nii.gz")

        nib.save(image_nifti, trg_image_path)
        nib.save(gt_nifti, trg_gt_path)

        image_paths.append(trg_image_path)
        gt_paths.append(trg_gt_path)

    return image_paths, gt_paths


def get_toothfairy_dataset(path, patch_shape, download=False, **kwargs):
    """Canal segmentation in CBCT
    https://toothfairy.grand-challenge.org/
    """
    image_paths, gt_paths = _get_toothfairy_paths(path, download)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        is_seg_dataset=True,
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_toothfairy_loader(path, patch_shape, batch_size, download=False, **kwargs):
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_toothfairy_dataset(path, patch_shape, download, **ds_kwargs)
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
