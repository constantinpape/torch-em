"""The BONBID-HIE dataset contains annotations for lesion segmentation in neonatal brain
diffusion MRI of patients with hypoxic-ischemic encephalopathy (HIE).

The dataset is released as part of the BONBID-HIE 2023 MICCAI challenge
(https://bonbid-hie2023.grand-challenge.org) and is hosted at
https://doi.org/10.5281/zenodo.10602767. It provides 85 training, 4 validation and 44
test cases (see `SPLITS`). Each case has a skull-stripped apparent diffusion coefficient
(ADC) map and the derived Z-ADC map (the ADC map normalized against a healthy reference
population, smoothed and clipped to the range [-6, 10]), both registered to a binary
lesion segmentation mask.

NOTE: The scans are distributed as MetaImage (.mha) files. This module converts them to
compressed nifti volumes on first use, which requires the SimpleITK python package.

NOTE: The Zenodo record's structured license field lists CC-BY-NC-ND 2.5, even though the
record's description text states that the data is released under the CC BY 4.0 license.
Please check the record for the authoritative license terms before further use.

This dataset is from the publication https://doi.org/10.1038/s41597-024-03986-7.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Optional, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "train": "https://zenodo.org/records/10602767/files/BONBID2023_Train.zip",
    "val": "https://zenodo.org/records/10602767/files/BONBID2023_Val.zip",
    "test": "https://zenodo.org/records/10602767/files/BONBID2023_Test.zip",
}

CHECKSUMS = {
    "train": "f1058093887daea1ebc07368b0642d3c40a3f70da5e2e225558ec32de3fa345b",
    "val": "8eaaa05cad00d5d5112583ca01be9ec685e15ca87b946b19a4010b3da35f4786",
    "test": "ac3988c57f035ee74a20dd70194e6fd51fe9aecc3a68e093f487620b24fcbfa3",
}

SPLITS = ("train", "val", "test")


def _convert_mha_to_nifti(mha_path, nifti_path):
    if os.path.exists(nifti_path):
        return

    import SimpleITK as sitk

    volume = sitk.ReadImage(mha_path)
    sitk.WriteImage(volume, nifti_path, useCompression=True)


def _preprocess_bonbid_hie(raw_dir, preprocessed_dir, split):
    adc_paths = natsorted(glob(os.path.join(raw_dir, "**", "1ADC_ss", "*-ADC_ss.mha"), recursive=True))
    if len(adc_paths) == 0:
        raise RuntimeError(f"Did not find any '{split}' scans at '{raw_dir}'.")

    for folder in ["adc", "zadc", "labels"]:
        os.makedirs(os.path.join(preprocessed_dir, folder), exist_ok=True)

    for adc_path in tqdm(adc_paths, desc=f"Preprocess BONBID-HIE '{split}' scans"):
        case_dir = os.path.dirname(os.path.dirname(adc_path))
        case_id = os.path.basename(adc_path)[:-len("-ADC_ss.mha")]

        zadc_path = os.path.join(case_dir, "2Z_ADC", f"Zmap_{case_id}-ADC_smooth2mm_clipped10.mha")
        label_path = os.path.join(case_dir, "3LABEL", f"{case_id}_lesion.mha")
        if not os.path.exists(zadc_path) or not os.path.exists(label_path):
            raise RuntimeError(f"Could not find the Z-ADC map or the label for the case '{case_id}'.")

        _convert_mha_to_nifti(adc_path, os.path.join(preprocessed_dir, "adc", f"{case_id}.nii.gz"))
        _convert_mha_to_nifti(zadc_path, os.path.join(preprocessed_dir, "zadc", f"{case_id}.nii.gz"))
        _convert_mha_to_nifti(label_path, os.path.join(preprocessed_dir, "labels", f"{case_id}.nii.gz"))


def get_bonbid_hie_data(path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False) -> str:  # noqa
    """Download the BONBID-HIE dataset and preprocess it to nifti volumes.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the preprocessed data is stored.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose from {SPLITS}.")

    preprocessed_dir = os.path.join(path, "preprocessed", split)
    if os.path.exists(preprocessed_dir):
        return preprocessed_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"BONBID2023_{split.capitalize()}.zip")
    util.download_source(path=zip_path, url=URLS[split], download=download, checksum=CHECKSUMS[split])

    raw_dir = os.path.join(path, "raw", split)
    util.unzip(zip_path=zip_path, dst=raw_dir)

    _preprocess_bonbid_hie(raw_dir, preprocessed_dir, split)

    return preprocessed_dir


def get_bonbid_hie_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    modality: Optional[Literal["adc", "zadc"]] = None,
    download: bool = False,
) -> Tuple[List, List[str]]:
    """Get paths to the BONBID-HIE data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        modality: The choice of modality for MRIs. Either 'adc' or 'zadc'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    preprocessed_dir = get_bonbid_hie_data(path, split, download)

    label_paths = natsorted(glob(os.path.join(preprocessed_dir, "labels", "*.nii.gz")))
    adc_paths = natsorted(glob(os.path.join(preprocessed_dir, "adc", "*.nii.gz")))
    zadc_paths = natsorted(glob(os.path.join(preprocessed_dir, "zadc", "*.nii.gz")))

    if modality is None:
        image_paths = [(adc_path, zadc_path) for adc_path, zadc_path in zip(adc_paths, zadc_paths)]
    elif modality == "adc":
        image_paths = adc_paths
    elif modality == "zadc":
        image_paths = zadc_paths
    else:
        raise ValueError(f"'{modality}' is not a valid modality.")

    return image_paths, label_paths


def get_bonbid_hie_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"] = "train",
    modality: Optional[Literal["adc", "zadc"]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BONBID-HIE dataset for segmentation of HIE-related brain lesions.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        modality: The choice of modality for MRIs. Either 'adc' or 'zadc'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_bonbid_hie_paths(path, split, modality, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        with_channels=modality is None,
        is_seg_dataset=True,
        **kwargs
    )


def get_bonbid_hie_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"] = "train",
    modality: Optional[Literal["adc", "zadc"]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BONBID-HIE dataloader for segmentation of HIE-related brain lesions.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        modality: The choice of modality for MRIs. Either 'adc' or 'zadc'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bonbid_hie_dataset(path, patch_shape, split, modality, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
