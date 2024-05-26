import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Optional

import numpy as np

import torch_em

from .. import util


URL = {
    "train": "https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip",
    "test": "https://zenodo.org/records/3431873/files/CHAOS_Test_Sets.zip"
}

CHECKSUM = {
    "train": "535f7d3417a0e0f0d9133fb3d962423d2a9cf3f103e4f09a3d8a1daf87d5d2fc",
    "test": "80e9e4d4c4e363f142de4570e9b698e3f92dcb5140cc25a9c1cf4963e5ae7541"
}


def get_chaos_data(path, split, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "data", "Train_Sets" if split == "train" else "Test_Sets")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, f"chaos_{split}.zip")
    util.download_source(path=zip_path, url=URL[split], download=download, checksum=CHECKSUM[split])
    util.unzip(zip_path=zip_path, dst=os.path.join(path, "data"))

    return data_dir


def _open_image(input_path):
    ext = os.path.splitext(input_path)[-1]

    if ext == ".dcm":
        import pydicom as dicom
        inputs = dicom.dcmread(input_path)
        inputs = inputs.pixel_array

    elif ext == ".png":
        import imageio.v3 as imageio
        inputs = imageio.imread(input_path)

    else:
        raise ValueError

    return inputs


def _get_chaos_paths(path, split, modality, download):
    data_dir = get_chaos_data(path=path, split=split, download=download)

    if modality is None:
        modality = ["CT", "MRI"]
    else:
        if isinstance(modality, str):
            modality = [modality]

    image_paths, gt_paths = [], []
    for m in modality:
        if m.upper() == "CT":
            m = m.upper()
            image_exts = ["DICOM_anon/*"]
            gt_exts = ["Ground/*"]

        elif m.upper().startswith("MR"):
            m = "MR"
            image_exts = ["T1DUAL/DICOM_anon/InPhase/*", "T2SPIR/DICOM_anon/*"]
            gt_exts = ["T1DUAL/Ground/*", "T2SPIR/Ground/*"]

        else:
            raise ValueError

        series_uids = glob(os.path.join(data_dir, m, "*"))

        for uid in tqdm(series_uids):
            _id = os.path.split(uid)[-1]

            base_dir = os.path.join(data_dir, "preprocessed", m.upper())

            os.makedirs(os.path.join(base_dir, "image"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "ground_truth"), exist_ok=True)

            for image_ext, gt_ext in zip(image_exts, gt_exts):
                if m == "MR":
                    modname = image_ext.split("/")[0] + "_MR"
                else:
                    modname = m

                image_path = os.path.join(base_dir, "image", f"{_id}_{modname}.nii.gz")
                gt_path = os.path.join(base_dir, "ground_truth", f"{_id}_{modname}.nii.gz")

                image_paths.append(image_path)
                gt_paths.append(gt_path)

                if os.path.exists(image_path) and os.path.exists(gt_path):
                    continue

                raw_slices = natsorted(glob(os.path.join(uid, image_ext)))
                gt_slices = natsorted(glob(os.path.join(uid, gt_ext)))

                raw = np.stack([_open_image(raw_slice) for raw_slice in raw_slices])
                gt = np.stack([_open_image(gt_slice) for gt_slice in gt_slices]).astype("uint8")

                raw = raw.transpose(1, 2, 0)
                gt = gt.transpose(1, 2, 0)

                import nibabel as nib
                raw_nifti = nib.Nifti2Image(raw, np.eye(4))
                nib.save(raw_nifti, image_path)

                gt_nifti = nib.Nifti2Image(gt, np.eye(4))
                nib.save(gt_nifti, gt_path)

    return image_paths, gt_paths


def get_chaos_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: str = "train",
    modality: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of abdominal organs in CT and MRI scans.

    This dataset is from Kavur et al. - https://doi.org/10.1016/j.media.2020.101950
    Please cite it if you use this dataset for a publication.
    """
    assert split == "train", "'train' is the only split with ground truth annotations."

    image_paths, gt_paths = _get_chaos_paths(path=path, split=split, modality=modality, download=download)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_chaos_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    split: str = "train",
    modality: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataloader for segmentation of abdominal organs in CT and MRI scans. See `get_chaos_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_chaos_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        modality=modality,
        resize_inputs=resize_inputs,
        download=download,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
