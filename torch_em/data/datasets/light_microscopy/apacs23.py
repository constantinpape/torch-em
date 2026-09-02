"""The APACS23 dataset contains annotations for cell segmentation in
digitized Pap smear images of the cervix.

APACS23 stands for Annotated PAp smear images for Cell Segmentation 2023. The dataset holds 3535
image and mask pairs of 2000 x 2000 pixels, which the authors split into a training and a test part.

NOTE: The masks are binary. They separate the cells from the background, and they hold no instance
id and no class. The publication reports about 37000 segmented cells, because the annotators
outlined every cell by hand, but the released masks merge them into one foreground.

NOTE: The repository stores every file on its own, and it offers no archive. The loader therefore
downloads about 7000 files, which takes a while on the first call. It stores them under `path`, so
a later call reads them from disk.

NOTE: A few files have no partner. The training part holds 10 images without a mask and 10 masks
without an image, and the test part holds 5 of each. The loader skips them, so it yields 2207
training pairs and 1328 test pairs.

The dataset is located at https://doi.org/10.17605/OSF.IO/CKA2F under the CC BY 4.0 license.
This dataset is from the publication https://doi.org/10.1038/s41597-024-03566-9.
Please cite it if you use this dataset in your research.
"""

import os
import time
import json
import urllib.error
import urllib.request
from glob import glob
from natsort import natsorted
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from typing import List, Literal, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


OSF_NODE = "cka2f"
OSF_API = f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/osfstorage/"

# The repository limits how many requests it serves per minute, so the download waits between files.
REQUEST_DELAY = 0.3
RETRY_DELAY = 5.0
MAX_RETRIES = 6

# The folder of a split, and the folders that hold its images and its masks.
SPLITS = {
    "train": ("training", "APACS23_Training_Input", "APACS23_Training_GroundTruth"),
    "test": ("test", "APACS23_Test_Input", "APACS23_Test_GroundTruth"),
}


def _list_folder(url: str) -> List[dict]:
    """List a folder of the repository, and follow its pages."""
    parts = urlparse(url)
    query = parse_qs(parts.query)
    query["page[size]"] = ["100"]
    url = urlunparse(parts._replace(query=urlencode(query, doseq=True)))

    entries, seen = [], set()
    while url:
        with urllib.request.urlopen(url, timeout=120) as response:
            page = json.load(response)
        for entry in page["data"]:
            # A page can repeat an entry, so the id decides whether it is new.
            if entry["id"] not in seen:
                seen.add(entry["id"])
                entries.append(entry)
        url = page["links"].get("next")
    return entries


def _download_file(url: str, output_path: str) -> None:
    """Download one file, and wait when the repository refuses the request.

    The repository answers with the status 403 once too many requests arrive in a short time, so
    every failed try waits longer than the one before it.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "torch-em"})
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                content = response.read()
            with open(output_path, "wb") as f:
                f.write(content)
            return
        except urllib.error.HTTPError as error:
            if error.code != 403 or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_DELAY * 2 ** attempt)

    raise RuntimeError(f"Could not download {url}.")


def _download_folder(entry: dict, output_dir: str) -> None:
    """Download every file of one repository folder."""
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)
    files = _list_folder(entry["relationships"]["files"]["links"]["related"]["href"])

    for item in tqdm(files, desc=f"Download '{os.path.basename(output_dir)}'"):
        name = item["attributes"]["name"]
        output_path = os.path.join(output_dir, name)
        if os.path.exists(output_path):
            continue
        _download_file(item["links"]["download"], output_path)
        time.sleep(REQUEST_DELAY)


def get_apacs23_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"] = "train",
    download: bool = False,
) -> Tuple[str, str]:
    """Download the APACS23 dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the images.
        The filepath to the folder with the masks.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(SPLITS)}.")

    folder_name, image_folder, label_folder = SPLITS[split]
    image_dir = os.path.join(path, split, "images")
    label_dir = os.path.join(path, split, "masks")
    if os.path.exists(image_dir) and os.path.exists(label_dir):
        return image_dir, label_dir

    if not download:
        raise RuntimeError(f"Cannot find the data at {os.path.join(path, split)}, but download was set to False.")

    os.makedirs(path, exist_ok=True)
    top = _list_folder(OSF_API)
    split_entry = next((e for e in top if e["attributes"]["name"] == folder_name), None)
    if split_entry is None:
        raise RuntimeError(f"Could not find the folder '{folder_name}' in the APACS23 repository.")

    inner = _list_folder(split_entry["relationships"]["files"]["links"]["related"]["href"])
    for entry in inner:
        name = entry["attributes"]["name"]
        if name == image_folder:
            _download_folder(entry, image_dir)
        elif name == label_folder:
            _download_folder(entry, label_dir)

    return image_dir, label_dir


def get_apacs23_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"] = "train",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the APACS23 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    image_dir, label_dir = get_apacs23_data(path, split, download)

    image_paths, label_paths = [], []
    for image_path in natsorted(glob(os.path.join(image_dir, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(label_dir, f"{stem}.png")
        # A few images have no mask, and a few masks have no image.
        if not os.path.exists(label_path):
            continue
        image_paths.append(image_path)
        label_paths.append(label_path)

    if not image_paths:
        raise RuntimeError(f"Could not find any APACS23 data in {image_dir}.")

    return image_paths, label_paths


def get_apacs23_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the APACS23 dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The APACS23 patch shape must be two-dimensional, got {patch_shape}.")

    image_paths, label_paths = get_apacs23_paths(path, split, download)
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        ndim=2,
        **kwargs,
    )


def get_apacs23_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the APACS23 dataloader for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_apacs23_dataset(
        path=path, patch_shape=patch_shape, split=split, download=download, **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
