import os
from glob import glob
from typing import Tuple, Optional

import torch

import torch_em

from .. import util


AUTOPET_DATA = "http://193.196.20.155/data/autoPET/data/nifti.zip"
CHECKSUM = "0ac2186ea6d936ff41ce605c6a9588aeb20f031085589897dbab22fc82a12972"


def _assort_autopet_dataset(path, download):
    target_dir = os.path.join(path, "AutoPET-II")
    if os.path.exists(target_dir):
        return

    os.makedirs(target_dir)
    zip_path = os.path.join(path, "autopet.zip")
    print("The AutoPET data is not available yet and will be downloaded.")
    print("Note that this dataset is large, so this step can take several hours (depending on your internet).")
    util.download_source(path=zip_path, url=AUTOPET_DATA, download=download, checksum=CHECKSUM)
    util.unzip(zip_path, target_dir, remove=False)


def _get_paths(path, modality):
    root_dir = os.path.join(path, "AutoPET-II", "FDG-PET-CT-Lesions", "*", "*")
    ct_paths = sorted(glob(os.path.join(root_dir, "CTres.nii.gz")))
    pet_paths = sorted(glob(os.path.join(root_dir, "SUV.nii.gz")))
    label_paths = sorted(glob(os.path.join(root_dir, "SEG.nii.gz")))
    if modality is None:
        raw_paths = [(ct_path, pet_path) for ct_path, pet_path in zip(ct_paths, pet_paths)]
    elif modality == "CT":
        raw_paths = ct_paths
    elif modality == "PET":
        raw_paths = pet_paths
    else:
        raise ValueError(f"Choose from the available modalities: `CT` / `PET`")

    return raw_paths, label_paths


def get_autopet_dataset(
    path: str,
    patch_shape: Tuple[int, ...],
    ndim: int,
    modality: Optional[str] = None,
    download: bool = False,
    **kwargs
) -> torch.utils.data.Dataset:
    """Dataset for lesion segmentation in whole-body FDG-PET/CT scans.

    This dataset is fromt the `AutoPET II - Automated Lesion Segmentation in PET/CT - Domain Generalization` challenge.
    Link: https://autopet-ii.grand-challenge.org/
    Please cite it if you use this dataset for publication.

    Arguments:
        path: The path where the zip files / the prepared dataset exists.
            - Expected initial structure: `path` should have ...
        patch_shape: The patch shape (for 2d or 3d patches)
        ndim: The dimensions of the inputs (use `2` for getting 2d patches, and `3` for getting 3d patches)
        modality: The modality for using the AutoPET dataset.
            - (default: None) If passed `None`, it takes both the modalities as inputs
        download: Downloads the dataset

    Returns:
        dataset: The segmentation dataset for the respective modalities.
    """ 
    _assort_autopet_dataset(path, download)
    raw_paths, label_paths = _get_paths(path, modality)
    dataset = torch_em.default_segmentation_dataset(
        raw_paths, "data", label_paths, "data",
        patch_shape, ndim=ndim, with_channels=modality is None,
        **kwargs
    )
    return dataset


def get_autopet_loader(
    path, patch_shape, batch_size, ndim, modality=None, download=False, **kwargs
):
    """Dataloader for lesion segmentation in whole-body FDG-PET/CT scans. See `get_autopet_dataset` for details."""
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_autopet_dataset(path, patch_shape, ndim, modality, download, **ds_kwargs)
    loader = torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
    return loader
