"""The PLETHORA dataset contains annotations for thoracic organs and pleural effusion in CT.

This dataset is from the publication https://doi.org/10.1002/mp.14424/.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from urllib.parse import urljoin
from typing import Union, Tuple, Literal, List

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

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
        "thoracic": "6dfcb60e46c7b0ccf240bc5d13acb1c45c8d2f4922223f7b2fbd5e37acff2be0",
        "pleural_effusion": "5dd07c327fb5723c5bbb48f2a02d7f365513d3ad136811fbe4def330ef2d7f6a"
    }
}


ZIPFILES = {
    "thoracic": "thoracic.zip",
    "pleural_effusion": "pleural_effusion.zip"
}


def get_plethora_data(
    path: Union[os.PathLike, str], task: Literal["thoracic", "pleural_effusion"], download: bool = False
) -> Tuple[str, str, str]:
    """Get the PLETHORA dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        task: The choice of task.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    image_dir = os.path.join(path, "data", "images")
    gt_dir = os.path.join(path, "data", "gt", "Thoracic_Cavities" if task == "thoracic" else "Effusions")
    csv_path = os.path.join(path, "plethora_images")
    if os.path.exists(image_dir) and os.path.exists(gt_dir):
        return image_dir, gt_dir, Path(csv_path).with_suffix(".csv")

    os.makedirs(path, exist_ok=True)

    # let's download dicom files from the tcia manifest
    tcia_path = os.path.join(path, "NSCLC-Radiomics-OriginalCTs.tcia")
    util.download_source_tcia(path=tcia_path, url=URL["image"], dst=image_dir, csv_filename=csv_path, download=download)

    # let's download the segmentations from zipfiles
    zip_path = os.path.join(path, ZIPFILES[task])
    util.download_source(
        path=zip_path, url=URL["gt"][task], download=download, checksum=CHECKSUMS["gt"][task]
    )
    util.unzip(zip_path=zip_path, dst=os.path.join(path, "data", "gt"))

    return image_dir, gt_dir, Path(csv_path).with_suffix(".csv")


def _assort_plethora_inputs(image_dir, gt_dir, task, csv_path):
    import nibabel as nib
    import pydicom as dicom

    df = pd.read_csv(csv_path)

    task_gt_dir = os.path.join(gt_dir, )

    os.makedirs(os.path.join(image_dir, "preprocessed"), exist_ok=True)
    os.makedirs(os.path.join(task_gt_dir, "preprocessed"), exist_ok=True)

    # let's get all the series uid of the volumes downloaded and spot their allocated subject id
    all_series_uid_dirs = glob(os.path.join(image_dir, "1.3*"))
    image_paths, gt_paths = [], []
    for series_uid_dir in tqdm(all_series_uid_dirs):
        series_uid = os.path.split(series_uid_dir)[-1]
        subject_id = pd.Series.to_string(df.loc[df["Series UID"] == series_uid]["Subject ID"])[-9:]

        try:
            gt_path = glob(os.path.join(task_gt_dir, subject_id, "*.nii.gz"))[0]
        except IndexError:
            # - some patients do not have "Thoracic_Cavities" segmentation
            print(f"The ground truth is missing for subject '{subject_id}'")
            continue

        assert os.path.exists(gt_path)

        vol_path = os.path.join(image_dir, "preprocessed", f"{subject_id}.nii.gz")
        neu_gt_path = os.path.join(task_gt_dir, "preprocessed", os.path.split(gt_path)[-1])

        image_paths.append(vol_path)
        gt_paths.append(neu_gt_path)
        if os.path.exists(vol_path) and os.path.exists(neu_gt_path):
            continue

        # the individual slices for the inputs need to be merged into one volume.
        if not os.path.exists(vol_path):
            all_dcm_slices = natsorted(glob(os.path.join(series_uid_dir, "*.dcm")))
            all_slices = []
            for dcm_path in all_dcm_slices:
                dcmfile = dicom.dcmread(dcm_path)
                img = dcmfile.pixel_array
                all_slices.append(img)

            volume = np.stack(all_slices)
            volume = volume.transpose(1, 2, 0)
            nii_vol = nib.Nifti1Image(volume, np.eye(4))
            nii_vol.header.get_xyzt_units()
            nii_vol.to_filename(vol_path)

        # the ground truth needs to be aligned as the inputs, let's take care of that.
        gt = nib.load(gt_path)
        gt = gt.get_fdata()
        gt = gt.transpose(2, 1, 0)  # aligning w.r.t the inputs
        gt = np.flip(gt, axis=(0, 1))

        gt = gt.transpose(1, 2, 0)
        gt_nii_vol = nib.Nifti1Image(gt, np.eye(4))
        gt_nii_vol.header.get_xyzt_units()
        gt_nii_vol.to_filename(neu_gt_path)

    return image_paths, gt_paths


def get_plethora_paths(
    path: Union[os.PathLike, str], task: Literal["thoracic", "pleural_effusion"], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the PLETHORA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        task: The choice of task.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    image_dir, gt_dir, csv_path = get_plethora_data(path, task, download)
    image_paths, gt_paths = _assort_plethora_inputs(image_dir=image_dir, gt_dir=gt_dir, task=task, csv_path=csv_path)
    return image_paths, gt_paths


def get_plethora_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    task: Literal["thoracic", "pleural_effusion"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PLETHORA dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        task: The choice of task.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_plethora_paths(path, task, download)

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


def get_plethora_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    task: Literal["thoracic", "pleural_effusion"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PLETHORA dataloader.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        task: The choice of task.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_plethora_dataset(path, patch_shape, task, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
