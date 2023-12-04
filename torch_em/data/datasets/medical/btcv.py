import os
from glob import glob
from typing import Optional, List, Tuple

from .. import util

import torch

import torch_em


_PATHS = {
    "Abdomen": "RawData.zip",
    "Cervix": "CervixRawData.zip"
}


# https://www.synapse.org/#!Synapse:syn3193805/wiki/217789
ABDOMEN_ORGANS = {
    "spleen": 1, "right kidney": 2, "left kidney": 3, "gallbladder": 4, "esophagus": 5, "liver": 6, "stomach": 7,
    "aorta": 8, "inferior vena cava": 9, "portal vein and splenic vein": 10, "pancreas": 11, "right adrenal gland": 12,
    "left adrenal gland": 13,
}


# https://www.synapse.org/#!Synapse:syn3193805/wiki/217790
CERVICAL_ORGANS = {
    "bladder": 1, "uterus": 2, "rectum": 3, "small bowel": 4
}


def _unzip_btcv_data(path, region):
    _target_dir = os.path.join(path, region)
    zip_path = os.path.join(path, _PATHS[region])

    # if the directory exists, we assume the assorting has been done
    if os.path.exists(_target_dir):
        return

    # now, let's prepare the directories where we unzip and store the inputs
    os.makedirs(_target_dir)

    # let's unzip the objects to the desired directory
    assert os.path.exists(zip_path), f"Looks like the zip file for {region} CT scans is missing."
    util.unzip(
        zip_path, _target_dir, remove=False
    )


def _assort_btcv_dataset(path, anatomy):
    if isinstance(anatomy, str):
        assert anatomy in _PATHS.keys(), anatomy
        _unzip_btcv_data(path, anatomy)

    elif isinstance(anatomy, list):
        for _region in anatomy:
            assert _region in _PATHS.keys(), anatomy
            _unzip_btcv_data(path, _region)

    else:
        raise TypeError


# TODO: we need to return specific labels, not 100% sure how to implement it
def _check_organ_match_anatomy(organs, anatomy):
    # if passed None, we return all organ labels
    if organs is None:
        return

    for organ in organs:
        if anatomy == "Abdomen":
            assert organ in ABDOMEN_ORGANS.keys(), f"{organ} not in {ABDOMEN_ORGANS.keys()}"
        elif anatomy == "Cervix":
            assert organ in CERVICAL_ORGANS.keys(), f"{organ} not in {CERVICAL_ORGANS.keys()}"


def _get_raw_and_label_paths(path, organs, anatomy):
    if isinstance(anatomy, str):
        _check_organ_match_anatomy(organs, anatomy)
        raw_paths = sorted(glob(os.path.join(path, anatomy, "Training", "img", "*.nii.gz")))
        label_paths = sorted(glob(os.path.join(path, anatomy, "Training", "label", "*.nii.gz")))

    elif isinstance(anatomy, list):
        breakpoint()
        raw_paths, label_paths = [], []
        for _region in anatomy:
            _check_organ_match_anatomy(organs, _region)
            all_raw_paths = sorted(glob(os.path.join(path, _region, "RawData", "Training", "img", "*.nii.gz")))
            all_label_paths = sorted(glob(os.path.join(path, _region, "RawData", "Training", "label", "*.nii.gz")))
            for tmp_raw_path, tmp_label_path in zip(all_raw_paths, all_label_paths):
                raw_paths.append(tmp_raw_path)
                label_paths.append(tmp_label_path)


def get_btcv_dataset(
        path: str,
        patch_shape: Tuple[int, ...],
        ndim: int,
        organs: Optional[List] = None,
        anatomy: Optional[List] = None,
        download: bool = False,
        **kwargs
) -> torch.utils.data.Dataset:
    """Dataset for multi-organ segmentation in CT scans.

    This dataset is from the Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge
    Link: https://www.synapse.org/#!Synapse:syn3193805/wiki/89480
    Please cite it if you use this dataset for a publication.

    Steps on how to get the dataset?
        1. Join the challenge using their official website: https://www.synapse.org/#!Synapse:syn3193805
        2. Next, go to "Files" -> (download the respective zip files)
            - "Abdomen" -> "RawData.zip" downloads all the abdominal CT scans
            - "Cervix" -> "CervixRawData.zip" downloads all the cervical CT scans
        3. Provide the path to the parent directory, where the zipped file(s) have been downloaded.
           The dataset would take care of the rest.

    Args:
        path: The path where the zip files / the prepared datasets exist.
            - Expected initial structure: `path` should have two zip files, namely `RawData.zip` and `CervixRawData.zip`
        patch_shape: The patch shape (for 2d or 3d patches)
        ndim: The dimensions of the inputs (use `2` for getting 2d patches,  and `3` for getting 3d patches)
        organ: The organs in the respective anatomical regions of choice
            - default: None (i.e., returns labels with all organ types)
        anatomy: The anatomical regions of choice from the provided scans
            - default: None (i.e., returns both the available anatomies - abdomen and cervix)
        download: (NOT SUPPORTED) Downloads the dataset

    Returns:
        dataset: The dataset for the respective splits
    """
    if download:
        raise NotImplementedError(
            "The BTCV dataset cannot be automatically download from `torch_em`. \
            Please download the dataset (see `get_btcv_dataset` for the download steps) \
            and provide the parent directory where the zip files are stored."
        )

    _assort_btcv_dataset(path, anatomy)

    raw_paths, label_paths = _get_raw_and_label_paths(path, organs, anatomy)
    assert len(raw_paths) == len(label_paths)

    return torch_em.default_segmentation_dataset(
        raw_paths, "data", label_paths, "data", patch_shape, ndim=ndim, **kwargs
    )


def get_btcv_loader(
        path,
        patch_shape,
        batch_size,
        ndim,
        organs=None,
        anatomy=None,
        download=False,
        **kwargs
):
    """Dataloader for multi-organ segmentation in CT scans.  See `get_btcv_dataset` for details."""
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_btcv_dataset(path, patch_shape, ndim, organs, anatomy, download, **ds_kwargs)
    loader = torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
    return loader
