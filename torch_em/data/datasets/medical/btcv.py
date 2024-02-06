import os
from glob import glob
from typing import Optional, List, Tuple

import torch

import torch_em

from .. import util
from ... import ConcatDataset


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

    # if the directory exists, we assume the assorting has been done
    if os.path.exists(_target_dir):
        return

    # now, let's prepare the directories where we unzip and store the inputs
    os.makedirs(_target_dir)

    # let's unzip the objects to the desired directory
    zip_path = os.path.join(path, _PATHS[region])
    assert os.path.exists(zip_path), f"Looks like the zip file for {region} CT scans is missing."
    util.unzip(zip_path, _target_dir, remove=False)


def _assort_btcv_dataset(path, anatomy):
    if anatomy is None:  # if not specified, we take both the anatomies into account
        anatomy = list(_PATHS.keys())

    if isinstance(anatomy, str):
        anatomy = [anatomy]

    for _region in anatomy:
        assert _region in _PATHS.keys(), anatomy
        _unzip_btcv_data(path, _region)

    return anatomy


def _check_organ_match_anatomy(organs, anatomy):
    # the sequence of anatomies assorted are:
    # we have a list of two list. list at first index is for abdomen, and second is for cervix
    from collections import defaultdict
    all_organs = defaultdict(list)
    if organs is None:  # if passed None, we return all organ labels
        if "Abdomen" in anatomy:
            all_organs["Abdomen"].append(list(ABDOMEN_ORGANS.keys()))

        if "Cervix" in anatomy:
            all_organs["Cervix"].append(list(CERVICAL_ORGANS.keys()))

        return all_organs

    if isinstance(organs, str):
        organs = [organs]

    for organ_name in organs:
        _match_found = False
        if organ_name in ABDOMEN_ORGANS and "Abdomen" in anatomy:
            all_organs["Abdomen"].append(organ_name)
            _match_found = True

        if organ_name in CERVICAL_ORGANS and "Cervix" in anatomy:
            all_organs["Cervix"].append(organ_name)
            _match_found = True

        if not _match_found:
            raise ValueError(f"{organ_name} not in {anatomy}")

    return all_organs


def _get_organ_ids(anatomy, organs):
    # now, let's get the organ ids
    for _region in anatomy:
        _region_dict = ABDOMEN_ORGANS if _region == "Abdomen" else CERVICAL_ORGANS
        per_region_organ_ids = [
            _region_dict[organ_name] for organ_name in organs[_region]
        ]
        organs[_region] = per_region_organ_ids

    return organs


def _get_raw_and_label_paths(path, anatomy):
    raw_paths, label_paths = {}, {}
    for _region in anatomy:
        raw_paths[_region] = sorted(glob(os.path.join(path, _region, "RawData", "Training", "img", "*.nii.gz")))
        label_paths[_region] = sorted(glob(os.path.join(path, _region, "RawData", "Training", "label", "*.nii.gz")))
    return raw_paths, label_paths


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

    anatomy = _assort_btcv_dataset(path, anatomy)
    organs = _check_organ_match_anatomy(organs, anatomy)
    organs = _get_organ_ids(anatomy, organs)
    raw_paths, label_paths = _get_raw_and_label_paths(path, anatomy)

    assert len(raw_paths) == len(label_paths)

    all_datasets = []
    for per_anatomy in anatomy:
        dataset = torch_em.default_segmentation_dataset(
            raw_paths[per_anatomy], "data",
            label_paths[per_anatomy], "data",
            patch_shape, ndim=ndim, semantic_ids=organs[per_anatomy],
            **kwargs
        )
        all_datasets.append(dataset)

    return ConcatDataset(*all_datasets)


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
    """Dataloader for multi-organ segmentation in CT scans. See `get_btcv_dataset` for details."""
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_btcv_dataset(path, patch_shape, ndim, organs, anatomy, download, **ds_kwargs)
    loader = torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
    return loader
