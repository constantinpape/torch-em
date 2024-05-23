import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple

import nrrd
import numpy as np
import nibabel as nib

import torch_em

from .. import util


URL = "https://zenodo.org/records/7442914/files/HaN-Seg.zip"
CHECKSUM = "20226dd717f334dc1b1afe961b3375f946fa56b64a80bf5349128f90c0bbfa5f"


def get_han_seg_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "HaN-Seg")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "HaN-Seg.zip")
    util.download_source(
        path=zip_path, url=URL, download=download, checksum=CHECKSUM
    )
    util.unzip(zip_path=zip_path, dst=path, remove=False)

    return data_dir


def _get_han_seg_paths(path, download):
    data_dir = get_han_seg_data(path=path, download=download)

    image_dir = os.path.join(data_dir, "set_1", "preprocessed", "images")
    gt_dir = os.path.join(data_dir, "set_1", "preprocessed", "ground_truth")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    image_paths, gt_paths = [], []
    all_case_dirs = natsorted(glob(os.path.join(data_dir, "set_1", "case_*")))
    for case_dir in tqdm(all_case_dirs):
        image_path = os.path.join(image_dir, f"{os.path.split(case_dir)[-1]}_ct.nii.gz")
        gt_path = os.path.join(gt_dir, f"{os.path.split(case_dir)[-1]}.nii.gz")
        image_paths.append(image_path)
        gt_paths.append(gt_path)
        if os.path.exists(image_path) and os.path.exists(gt_path):
            continue

        all_nrrd_paths = natsorted(glob(os.path.join(case_dir, "*.nrrd")))
        all_volumes, all_volume_ids = [], []
        for nrrd_path in all_nrrd_paths:
            image_id = Path(nrrd_path).stem

            # we skip the MRI volumes
            if image_id.endswith("_MR_T1"):
                continue

            data, header = nrrd.read(nrrd_path)
            all_volumes.append(data.transpose(2, 0, 1))
            all_volume_ids.append(image_id)

        raw = all_volumes[0]
        raw = nib.Nifti2Image(raw, np.eye(4))
        nib.save(raw, image_path)

        gt = np.zeros(raw.shape)
        for idx, per_organ in enumerate(all_volumes[1:], 1):
            gt[per_organ > 0] = idx
        gt = nib.Nifti2Image(gt, np.eye(4))
        nib.save(gt, gt_path)

    return image_paths, gt_paths


def get_han_seg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for head and neck organ-at-rish segmentation in CT scans.

    This dataset is from Podobnik et al. - https://doi.org/10.1002/mp.16197
    Please cite it if you use it in a publication.
    """
    image_paths, gt_paths = _get_han_seg_paths(path=path, download=download)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_han_seg_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for for head and neck organ-at-rish segmentation in CT scans. See `get_han_seg_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_han_seg_dataset(
        path=path, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
