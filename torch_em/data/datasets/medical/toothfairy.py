"""The ToothSeg data contains annotations for mandibular canal (v1) and multiple structures (v2)
segmentation in CBCT scans.

NOTE: The dataset is located at https://ditto.ing.unimore.it/
To download the dataset, please follow the mentioned steps:
- Choose either v1 (https://ditto.ing.unimore.it/toothfairy) or v2 (https://ditto.ing.unimore.it/toothfairy2).
- Visit the website, scroll down to the 'Download' section, which expects you to sign up.
- After signing up, use your credentials to login to the dataset home page.
- Click on the blue icon stating: 'Download Dataset' to download the zipped files to the desired path.

The relevant links for the dataset are:
- ToothFairy Challenge: https://toothfairy.grand-challenge.org/
- ToothFairy2 Challenge: https://toothfairy2.grand-challenge.org/
- Publication 1: https://doi.org/10.1109/ACCESS.2022.3144840
- Publication 2: https://doi.org/10.1109/CVPR52688.2022.02046

Please cite them if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def get_toothfairy_data(
    path: Union[os.PathLike, str], version: Literal["v1", "v2"] = "v2", download: bool = False
) -> str:
    """Obtain the ToothFairy datasets.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        version: The version of dataset. Either v1 (ToothFairy) or v2 (ToothFairy2).
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the already downloaded dataset.
    """
    data_dir = os.path.join(path, "ToothFairy_Dataset/Dataset" if version == "v1" else "Dataset112_ToothFairy2")
    if os.path.exists(data_dir):
        return data_dir

    if download:
        msg = "Download is set to True, but 'torch_em' cannot download this dataset. "
        msg += "See `get_toothfairy2_data` for details."
        raise NotImplementedError(msg)

    if version == "v1":
        zip_path = os.path.join(path, "ToothFairy_Dataset.zip")
    elif version == "v2":
        zip_path = os.path.join(path, "ToothFairy2_Dataset.zip")
    else:
        raise ValueError(f"'{version}' is not a valid version.")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"It's expected to place the downloaded toothfairy zipfile at '{path}'.")

    util.unzip(zip_path=zip_path, dst=path, remove=False)

    return data_dir


def _preprocess_toothfairy_inputs(path, data_dir):
    import nibabel as nib

    images_dir = os.path.join(path, "data", "images")
    gt_dir = os.path.join(path, "data", "dense_labels")
    if os.path.exists(images_dir) and os.path.exists(gt_dir):
        return natsorted(glob(os.path.join(images_dir, "*.nii.gz"))), natsorted(glob(os.path.join(gt_dir, "*.nii.gz")))

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    image_paths, gt_paths = [], []
    for patient_dir in tqdm(glob(os.path.join(data_dir, "P*")), desc="Preprocessing inputs"):
        dense_anns_path = os.path.join(patient_dir, "gt_alpha.npy")
        if not os.path.exists(dense_anns_path):
            continue

        image_path = os.path.join(patient_dir, "data.npy")
        image, gt = np.load(image_path), np.load(dense_anns_path)
        image_nifti, gt_nifti = nib.Nifti2Image(image, np.eye(4)), nib.Nifti2Image(gt, np.eye(4))

        patient_id = os.path.split(patient_dir)[-1]
        trg_image_path = os.path.join(images_dir, f"{patient_id}.nii.gz")
        trg_gt_path = os.path.join(gt_dir, f"{patient_id}.nii.gz")

        nib.save(image_nifti, trg_image_path)
        nib.save(gt_nifti, trg_gt_path)

        image_paths.append(trg_image_path)
        gt_paths.append(trg_gt_path)

    return image_paths, gt_paths


def get_toothfairy_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'val', 'test'],
    version: Literal["v1", "v2"] = "v2",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the ToothFairy data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        version: The version of dataset. Either 'v1' (ToothFairy) or 'v2' (ToothFairy2).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_toothfairy_data(path, version, download)

    if version == "v1":
        image_paths, gt_paths = _preprocess_toothfairy_inputs(path, data_dir)

        if split == "train":
            image_paths, gt_paths = image_paths[:100], gt_paths[:100]
        elif split == "val":
            image_paths, gt_paths = image_paths[100:125], gt_paths[100:125]
        elif split == "test":
            image_paths, gt_paths = image_paths[125:], gt_paths[125:]
        else:
            raise ValueError(f"'{split}' is not a valid split.")

    else:
        image_paths = natsorted(glob(os.path.join(data_dir, "imagesTr", "*.mha")))
        gt_paths = natsorted(glob(os.path.join(data_dir, "labelsTr", "*.mha")))

        if split == "train":
            image_paths, gt_paths = image_paths[:400], gt_paths[:400]
        elif split == "val":
            image_paths, gt_paths = image_paths[400:425], gt_paths[400:425]
        elif split == "test":
            image_paths, gt_paths = image_paths[425:], gt_paths[425:]
        else:
            raise ValueError(f"'{split}' is not a valid split.")

    return image_paths, gt_paths


def get_toothfairy_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    version: Literal["v1", "v2"] = "v2",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ToothFairy dataset for canal and teeth segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        version: The version of dataset. Either 'v1' (ToothFairy) or 'v2' (ToothFairy2).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_toothfairy_paths(path, split, version, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data" if version == "v1" else None,
        label_paths=gt_paths,
        label_key="data" if version == "v1" else None,
        is_seg_dataset=True,
        patch_shape=patch_shape,
        **kwargs
    )


def get_toothfairy_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    version: Literal["v1", "v2"] = "v2",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ToothFairy dataloader for canal and teeth segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        version: The version of dataset. Either 'v1' (ToothFairy) or 'v2' (ToothFairy2).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_toothfairy_dataset(path, patch_shape, split, version, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
