import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import nibabel as nib
import pydicom as dicom

import torch_em

from .. import util


BASE_URL = "https://wiki.cancerimagingarchive.net/download/attachments/68551327/"


URL = {
    "image": urljoin(BASE_URL, "NSCLC-Radiomics-OriginalCTs.tcia"),
    "gt": {
        "thoracic": urljoin(
            BASE_URL, "PleThora%20Thoracic_Cavities%20June%202020.zip?version=1&modificationDate=1593202695428&api=v2"
        ),
        "pleural_effusion": urljoin(
            BASE_URL, "PleThora%20Effusions%20June%202020.zip?version=1&modificationDate=1593202778373&api=v2"
        )
    }
}


CHECKSUMS = {
    "image": None,
    "gt": {
        "thoracic": None,
        "pleural_effusion": None
    }
}


ZIPFILES = {
    "thoracic": "thoracic.zip",
    "pleural_effusion": "pleural_effusion.zip"
}


def get_plethora_data(path, task, download):
    os.makedirs(path, exist_ok=True)

    image_dir = os.path.join(path, "data", "images")
    gt_dir = os.path.join(path, "data", "gt")
    csv_path = os.path.join(path, "plethora_images")
    if os.path.exists(image_dir) and os.path.exists(gt_dir):
        return image_dir, gt_dir, Path(csv_path).with_suffix(".csv")

    # let's download dicom files from the tcia manifest
    tcia_path = os.path.join(path, "NSCLC-Radiomics-OriginalCTs.tcia")
    util.download_source_tcia(path=tcia_path, url=URL, dst=image_dir, csv_filename=csv_path)

    # let's download the segmentations from zipfiles
    zip_path = os.path.join(path, ZIPFILES[task])
    util.download_source(
        path=zip_path, url=URL["gt"][task], download=download, checksum=CHECKSUMS["gt"][task]
    )
    util.unzip(zip_path=zip_path, dst=gt_dir)

    return image_dir, gt_dir, Path(csv_path).with_suffix(".csv")


def _assort_plethora_inputs(image_dir, gt_dir, task, csv_path):
    df = pd.read_csv(csv_path)

    task_gt_dir = os.path.join(gt_dir, "Thoracic_Cavity" if task == "thoracic" else "Pleural_Effusion")

    # let's get all the series uid of the volumes downloaded and spot their allocated subject id
    all_series_uid_dirs = glob(os.path.join(image_dir, "1.3*"))
    for series_uid_dir in all_series_uid_dirs:
        series_uid = os.path.split(series_uid_dir)[-1]
        subject_id = pd.Series.to_string(df.loc[df["Series UID"] == series_uid]["Subject ID"])[-9:]

        vol_path = os.path.join(image_dir, f"{subject_id}.nii.gz")
        if os.path.exists(vol_path):
            continue

        # TODO: there are multiple raters, check it out if there is can be some consistency
        gt_path = glob(os.path.join(task_gt_dir, subject_id, "*_primary_reviewer.nii.gz"))[0]

        # the ground truth needs to be aligned as the inputs, let's take care of that.
        gt = nib.load(gt_path)
        gt = gt.get_fdata()
        gt = gt.transpose(2, 1, 0)  # aligning w.r.t the inputs
        gt = np.flip(gt, axis=(0, 1))

        all_dcm_slices = natsorted(glob(os.path.join(series_uid_dir, "*.dcm")))
        all_slices = []
        for dcm_path in all_dcm_slices:
            dcmfile = dicom.dcmread(dcm_path)
            img = dcmfile.pixel_array
            all_slices.append(img)

        volume = np.stack(all_slices)
        nii_vol = nib.Nifti1Image(volume, np.eye(4))
        nii_vol.header.get_xyzt_units()
        nii_vol.to_filename(vol_path)


def _get_plethora_paths(path, task, download):
    image_dir, gt_dir, csv_path = get_plethora_data(path=path, task=task, download=download)

    _assort_plethora_inputs(image_dir=image_dir, gt_dir=gt_dir, task=task, csv_path=csv_path)

    image_paths = ...
    gt_paths = ...

    return image_paths, gt_paths


def get_plethora_dataset(
    path: Union[os.PathLike, str],
    task: str,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    image_paths, gt_paths = _get_plethora_paths(path=path, task=task, download=download)

    dataset = ...

    return dataset


def get_plethora_loader(
    path: Union[os.PathLike, str],
    task: str,
    patch_shape: Tuple[int, ...],
    batch_size: int,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_plethora_dataset(
        path=path, task=task, patch_shape=patch_shape, resize_inputs=resize_inputs, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
