"""The CuRIOUS2022 dataset contains annotations for brain tumor segmentation (before resection) and
resection cavity segmentation (after resection) in intra-operative brain ultrasound.

This is the segmentation track of the CuRIOUS 2022 MICCAI challenge, located at
https://curious2022.grand-challenge.org. The underlying ultrasound (and MRI) volumes are from the
RESECT database, located at https://doi.org/10.11582/2017.00004, and the segmentation annotations are
from the RESECT-SEG dataset, located at https://osf.io/jv8bk (published under CC-BY-NC-SA-4.0).
Please cite the following publications if you use this dataset in your research:
- Y. Xiao et al., "REtroSpective Evaluation of Cerebral Tumors (RESECT): A clinical database of
  pre-operative MRI and intra-operative ultrasound in low-grade glioma surgeries", Medical Physics, 2017.
  https://doi.org/10.1002/mp.12268
- B. Behboodi et al., "Open access segmentations of intraoperative brain tumor ultrasound images",
  Medical Physics, 2024. https://doi.org/10.1002/mp.17317
"""

import os
from typing import Union, Tuple, Literal, List

import requests

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


RESECT_DATASET_ID = "5686d8fa-2003-4837-8e66-8e887fabe21e"
RESECT_BASE_URL = f"https://data.archive.sigma2.no/dataset/{RESECT_DATASET_ID}/download/RESECT/NIFTI"

RESECT_SEG_NODE_ID = "jv8bk"
RESECT_SEG_ROOT_FOLDER_ID = "64cd20819cbf033b051e46c8"

CASE_IDS = [2, 3, 7, 8, 11, 12, 15, 17, 18, 23]
"""The RESECT case ids for which the RESECT-SEG dataset provides tumor and / or resection cavity
segmentations. Note that Case11 does not have a resection cavity annotation."""


def _osf_list_all(url):
    items = []
    while url:
        r = requests.get(url)
        r.raise_for_status()
        payload = r.json()
        items.extend(payload["data"])
        url = payload["links"].get("next")
    return items


def _get_seg_case_folder_ids():
    root_url = f"https://api.osf.io/v2/nodes/{RESECT_SEG_NODE_ID}/files/osfstorage/{RESECT_SEG_ROOT_FOLDER_ID}/"
    items = _osf_list_all(root_url)
    return {
        item["attributes"]["name"]: item["id"] for item in items if item["attributes"]["kind"] == "folder"
    }


def _get_seg_file_urls(folder_id):
    url = f"https://api.osf.io/v2/nodes/{RESECT_SEG_NODE_ID}/files/osfstorage/{folder_id}/"
    items = _osf_list_all(url)
    return {item["attributes"]["name"]: item["links"]["download"] for item in items}


def get_curious2022_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CuRIOUS2022 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    os.makedirs(path, exist_ok=True)

    case_folder_ids = None
    for case_id in CASE_IDS:
        case_dir = os.path.join(path, f"Case{case_id}")
        os.makedirs(case_dir, exist_ok=True)

        for stage in ["before", "after"]:
            fname = f"Case{case_id}-US-{stage}.nii.gz"
            dst = os.path.join(case_dir, fname)
            if os.path.exists(dst):
                continue
            url = f"{RESECT_BASE_URL}/Case{case_id}/US/{fname}"
            util.download_source(path=dst, url=url, download=download)

        for label_type, stage in [("tumor", "before"), ("resection", "after")]:
            fname = f"Case{case_id}-US-{stage}-{label_type}.nii.gz"
            dst = os.path.join(case_dir, fname)
            if os.path.exists(dst):
                continue
            if not download:
                # Case11 does not have a resection cavity annotation, so we cannot know without
                # querying the OSF API whether the file is expected to exist.
                continue

            if case_folder_ids is None:
                case_folder_ids = _get_seg_case_folder_ids()

            folder_id = case_folder_ids.get(f"Case{case_id}")
            if folder_id is None:
                continue

            file_urls = _get_seg_file_urls(folder_id)
            url = file_urls.get(fname)
            if url is None:  # e.g. Case11 has no resection cavity annotation.
                continue

            util.download_source(path=dst, url=url, download=download)

    return path


def get_curious2022_paths(
    path: Union[os.PathLike, str], task: Literal["tumor", "resection"] = "tumor", download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CuRIOUS2022 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        task: The choice of segmentation task. Either 'tumor' (before resection) or
            'resection' (resection cavity, after resection).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if task not in ["tumor", "resection"]:
        raise ValueError(f"'{task}' is not a valid task. Choose either 'tumor' or 'resection'.")

    get_curious2022_data(path, download)

    stage = "before" if task == "tumor" else "after"

    raw_paths, label_paths = [], []
    for case_id in CASE_IDS:
        case_dir = os.path.join(path, f"Case{case_id}")
        raw_path = os.path.join(case_dir, f"Case{case_id}-US-{stage}.nii.gz")
        label_path = os.path.join(case_dir, f"Case{case_id}-US-{stage}-{task}.nii.gz")
        if os.path.exists(raw_path) and os.path.exists(label_path):
            raw_paths.append(raw_path)
            label_paths.append(label_path)

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_curious2022_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    task: Literal["tumor", "resection"] = "tumor",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CuRIOUS2022 dataset for brain tumor / resection cavity segmentation in intra-operative
    brain ultrasound.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        task: The choice of segmentation task. Either 'tumor' (before resection) or
            'resection' (resection cavity, after resection).
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_curious2022_paths(path, task, download)

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


def get_curious2022_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    task: Literal["tumor", "resection"] = "tumor",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CuRIOUS2022 dataloader for brain tumor / resection cavity segmentation in intra-operative
    brain ultrasound.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        task: The choice of segmentation task. Either 'tumor' (before resection) or
            'resection' (resection cavity, after resection).
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the
            PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_curious2022_dataset(path, patch_shape, task, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
