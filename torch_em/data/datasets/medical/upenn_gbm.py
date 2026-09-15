"""The UPENN-GBM dataset contains annotations for brain tumor segmentation in multi-parametric MRI
of patients with de novo glioblastoma.

It consists of 671 skull-stripped and co-registered MRI scans (630 patients, 240 x 240 x 155 voxels at 1 mm
isotropic resolution in SRI24 atlas space) with the modalities T1, T1GD (post-contrast T1), T2 and FLAIR.
The tumor sub-regions are labeled following the BraTS convention: 1: necrotic and non-enhancing tumor core,
2: peritumoral edema, 4: GD-enhancing tumor.
Two kinds of segmentations are available: 147 scans have segmentations that were manually corrected and approved
by expert neuroradiologists ('manual', the default) and 611 scans have automatically generated segmentations
(label fusion of an ensemble of BraTS models, 'automated', which includes the 147 manually corrected scans).

The dataset is located at https://www.cancerimagingarchive.net/collection/upenn-gbm/ and the nifti files are only
offered via IBM Aspera there. We download the data from a mirror of the original nifti files at
https://huggingface.co/datasets/MedOtter/UPENN-GBM instead.

This dataset is from the publication https://doi.org/10.1038/s41597-022-01560-7.
The data was released at https://doi.org/10.7937/TCIA.709X-DN49.
Please cite it if you use this dataset in your research.
"""

import os
import json
from tqdm import tqdm
from typing import Union, Tuple, Literal, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://huggingface.co/datasets/MedOtter/UPENN-GBM/resolve/main/"

# The manifest lists all scans with their image and segmentation files. The nifti files are not checksummed.
CHECKSUM = "2b2c93a4181e9fc8e7db7d7b977cfb1981f63d13dcf19adc501df5cf92a70c76"

MODALITIES = ["T1", "T1GD", "T2", "FLAIR"]

LABEL_IDS = {"necrotic_core": 1, "edema": 2, "enhancing_tumor": 4}


def _get_scans(path, segmentation, download):
    manifest_path = os.path.join(path, "subjects_manifest.json")
    util.download_source(path=manifest_path, url=f"{URL}subjects_manifest.json", download=download, checksum=CHECKSUM)
    with open(manifest_path, "r") as f:
        scans = json.load(f)["subjects"]
    return [scan for scan in scans if scan[f"{segmentation}_segm"] is not None]


def get_upenn_gbm_data(
    path: Union[os.PathLike, str], segmentation: Literal["manual", "automated"] = "manual", download: bool = False
) -> str:
    """Download the UPENN-GBM dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        segmentation: The kind of segmentation. Either 'manual' (corrected by experts) or 'automated'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    if segmentation not in ("manual", "automated"):
        raise ValueError(f"'{segmentation}' is not a valid segmentation. Please choose 'manual' or 'automated'.")

    os.makedirs(path, exist_ok=True)
    scans = _get_scans(path, segmentation, download)

    for scan in tqdm(scans, desc=f"Download UPENN-GBM ({segmentation} segmentations)"):
        for rel_path in list(scan["modalities"].values()) + [scan[f"{segmentation}_segm"]]:
            fpath = os.path.join(path, rel_path)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            util.download_source(path=fpath, url=f"{URL}{rel_path}", download=download, checksum=None)

    return path


def get_upenn_gbm_paths(
    path: Union[os.PathLike, str],
    modality: Optional[Literal["T1", "T1GD", "T2", "FLAIR"]] = None,
    segmentation: Literal["manual", "automated"] = "manual",
    download: bool = False,
) -> Tuple[List[Union[str, Tuple[str, ...]]], List[str]]:
    """Get paths to the UPENN-GBM data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The choice of modality. One of 'T1', 'T1GD', 'T2' or 'FLAIR'.
            By default, all modalities are returned as channels.
        segmentation: The kind of segmentation. Either 'manual' (corrected by experts) or 'automated'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if modality is not None and modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Please choose from {MODALITIES}.")

    data_dir = get_upenn_gbm_data(path, segmentation, download)
    scans = _get_scans(data_dir, segmentation, download)

    if modality is None:
        raw_paths = [tuple(os.path.join(data_dir, scan["modalities"][mod]) for mod in MODALITIES) for scan in scans]
    else:
        raw_paths = [os.path.join(data_dir, scan["modalities"][modality]) for scan in scans]
    label_paths = [os.path.join(data_dir, scan[f"{segmentation}_segm"]) for scan in scans]

    return raw_paths, label_paths


def get_upenn_gbm_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Optional[Literal["T1", "T1GD", "T2", "FLAIR"]] = None,
    segmentation: Literal["manual", "automated"] = "manual",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the UPENN-GBM dataset for brain tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The choice of modality. One of 'T1', 'T1GD', 'T2' or 'FLAIR'.
            By default, all modalities are used as channels.
        segmentation: The kind of segmentation. Either 'manual' (corrected by experts) or 'automated'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_upenn_gbm_paths(path, modality, segmentation, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        with_channels=modality is None,
        **kwargs
    )


def get_upenn_gbm_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Optional[Literal["T1", "T1GD", "T2", "FLAIR"]] = None,
    segmentation: Literal["manual", "automated"] = "manual",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the UPENN-GBM dataloader for brain tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The choice of modality. One of 'T1', 'T1GD', 'T2' or 'FLAIR'.
            By default, all modalities are used as channels.
        segmentation: The kind of segmentation. Either 'manual' (corrected by experts) or 'automated'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_upenn_gbm_dataset(path, patch_shape, modality, segmentation, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
