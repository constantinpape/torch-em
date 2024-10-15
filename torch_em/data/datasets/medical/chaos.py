"""The CHAOS dataset contains annotations for segmentation of abdominal organs in
CT and MRI scans.

This dataset is from the publication ttps://doi.org/10.1016/j.media.2020.101950.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Optional, Literal, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

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


def get_chaos_data(
    path: Union[os.PathLike, str], split: Literal['train', 'test'] = "train", download: bool = False
) -> str:
    """Download the CHAOS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downlaoded.
    """
    assert split == "train", "'train' is the only split with ground truth annotations."

    data_dir = os.path.join(path, "data", "Train_Sets" if split == "train" else "Test_Sets")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

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


def _preprocess_inputs(data_dir, modality):
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


def get_chaos_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'test'] = "train",
    modality: Optional[Literal['CT', 'MRI']] = None,
    download: bool = False
) -> Tuple[List[int], List[int]]:
    """Get paths to the CHAOS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'train', or 'test'.
        modality: The choice of modality. Either 'CT' or 'MRI'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_chaos_data(path=path, split=split, download=download)

    if modality is None:
        modality = ["CT", "MRI"]
    else:
        if isinstance(modality, str):
            modality = [modality]

    image_paths, gt_paths = _preprocess_inputs(data_dir, modality)

    return image_paths, gt_paths


def get_chaos_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'test'] = "train",
    modality: Optional[Literal['CT', 'MRI']] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CHAOS dataset for abdominal organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The data split to use. Either 'train', or 'test'.
        modality: The choice of modality. Either 'CT' or 'MRI'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_chaos_paths(path, split, modality, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths, raw_key="data", label_paths=gt_paths, label_key="data", patch_shape=patch_shape, **kwargs
    )


def get_chaos_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    batch_size: int,
    split: str = "train",
    modality: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CHAOS dataloader for abdominal organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The data split to use. Either 'train', or 'test'.
        modality: The choice of modality. Either 'CT' or 'MRI'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_chaos_dataset(path, patch_shape, split, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
