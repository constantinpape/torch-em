"""The Tongue MRI dataset contains annotations for tongue musculature segmentation in MRI.

The dataset consists of 47 healthy subjects collated from three studies: BMC (24 subjects,
T1-weighted MRI), EATT4MND (19 subjects, T2-weighted MRI) and BeLong (4 subjects, T2-weighted
MRI). Each subject has a manually corrected semantic segmentation of four tongue muscles,
following `LABEL_IDS`: the genioglossus, the superior longitudinal, the inferior longitudinal,
and the combined transverse / vertical muscle.

The dataset also ships study templates for each site (in the 'template' folders next to the
'images' and 'labels' folders that this module uses), which are not returned by this module.

The dataset is located at https://osf.io/wt9fc/ and is distributed under the CC0 1.0 license.

This dataset is from the publication https://doi.org/10.1038/s41597-025-05092-8.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


# The site sub-folders are generated on-the-fly by OSF, hence the checksums of the archives are not reliable.
URLS = {
    "BMC": {
        "images": "https://files.osf.io/v1/resources/wt9fc/providers/osfstorage/66e224fb34c37b7bbdcb074e/?zip=",
        "labels": "https://files.osf.io/v1/resources/wt9fc/providers/osfstorage/66e224ff87c914991b64b5cb/?zip=",
    },
    "EATT4MND": {
        "images": "https://files.osf.io/v1/resources/wt9fc/providers/osfstorage/66e2259e2d9d10b9f9cb0ab2/?zip=",
        "labels": "https://files.osf.io/v1/resources/wt9fc/providers/osfstorage/66e225a28cec7392871d9456/?zip=",
    },
    "BeLong": {
        "images": "https://files.osf.io/v1/resources/wt9fc/providers/osfstorage/66e2248cf57074722a64c057/?zip=",
        "labels": "https://files.osf.io/v1/resources/wt9fc/providers/osfstorage/66e224931bc7424b3dc05a46/?zip=",
    },
}

SITES = list(URLS.keys())

LABEL_IDS = {
    "background": 0,
    "genioglossus": 1,
    "superior_longitudinal": 2,
    "inferior_longitudinal": 3,
    "transverse_vertical": 4,
}
"""The semantic label ids of the tongue muscle classes."""


def get_tongue_mri_data(
    path: Union[os.PathLike, str], site: Literal["BMC", "EATT4MND", "BeLong"] = "BMC", download: bool = False
) -> str:
    """Download the Tongue MRI dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        site: The study site to download. One of 'BMC', 'EATT4MND', 'BeLong'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the site folder where the data is downloaded.
    """
    if site not in SITES:
        raise ValueError(f"'{site}' is not a valid site. Choose one of {SITES}.")

    site_dir = os.path.join(path, site)
    os.makedirs(site_dir, exist_ok=True)

    for name in ["images", "labels"]:
        data_dir = os.path.join(site_dir, name)
        if os.path.exists(data_dir):
            continue

        zip_path = os.path.join(site_dir, f"{name}.zip")
        util.download_source(path=zip_path, url=URLS[site][name], download=download, checksum=None)
        util.unzip(zip_path=zip_path, dst=data_dir)

    return site_dir


def get_tongue_mri_paths(
    path: Union[os.PathLike, str],
    site: Optional[Literal["BMC", "EATT4MND", "BeLong"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Tongue MRI data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        site: The study site to use. One of 'BMC', 'EATT4MND', 'BeLong'. By default, all sites are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    sites = SITES if site is None else [site]

    raw_paths, label_paths = [], []
    for s in sites:
        site_dir = get_tongue_mri_data(path, s, download)
        cur_label_paths = natsorted(glob(os.path.join(site_dir, "labels", "*_labels.nii.gz")))
        cur_raw_paths = [
            os.path.join(site_dir, "images", os.path.basename(p).replace("_labels.nii.gz", ".nii.gz"))
            for p in cur_label_paths
        ]
        assert all(os.path.exists(p) for p in cur_raw_paths), f"Some image volumes are missing for site '{s}'."
        raw_paths.extend(cur_raw_paths)
        label_paths.extend(cur_label_paths)

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_tongue_mri_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    site: Optional[Literal["BMC", "EATT4MND", "BeLong"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Tongue MRI dataset for tongue musculature segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        site: The study site to use. One of 'BMC', 'EATT4MND', 'BeLong'. By default, all sites are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_tongue_mri_paths(path, site, download)

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
        **kwargs
    )


def get_tongue_mri_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    site: Optional[Literal["BMC", "EATT4MND", "BeLong"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Tongue MRI dataloader for tongue musculature segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        site: The study site to use. One of 'BMC', 'EATT4MND', 'BeLong'. By default, all sites are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_tongue_mri_dataset(path, patch_shape, site, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
