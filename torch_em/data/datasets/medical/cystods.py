"""The CystoDS dataset contains annotations for tumor, ureteral orifice, resection scar and air bubble
segmentation in white light cystoscopy images.

The full dataset consists of 8,067 images from 160 patients, labelled with five classes and
22 subclasses for bladder cancer detection. Of these, 768 images have additional pixel-level
segmentation annotations (stored as polygon shapes). This module exposes the 763 of these 768
images that are actually present in the 'images' folder of the OSF release (5 filenames are
listed with a segmentation in the metadata but are missing from the images folder, see
`MISSING_IMAGES`). The remaining, unsegmented images only have classification labels and are
out of scope for segmentation.

The dataset is located at https://osf.io/xvdhy (CC BY 4.0).
This dataset is from the publication https://doi.org/10.1038/s41597-026-06887-z.
Please cite it if you use this dataset in your research.
"""

import os
import csv
import json
from glob import glob
from warnings import warn
from tqdm import tqdm
from natsort import natsorted
from typing import List, Tuple, Union

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


OSF_NODE_ID = "xvdhy"
CSV_URL = "https://osf.io/download/n8cxz/"
CSV_CHECKSUM = "7988766bf2d17e7607ab2fcd9ad58d20a5fa4d7e107838a3b33b779e4e423f13"

IMAGES_FOLDER_ID = "6883c4851f60df395a84faee"
SEGMENTATIONS_FOLDER_ID = "688413af2a29f9bd26e7cc33"

LABEL_MAP = {
    "Tumor": 1,
    "Flat Tumor": 2,
    "Left ureteral orifice": 3,
    "Right ureteral orifice": 4,
    "Resection scar": 5,
    "Air bubble": 6,
}

# 5 of the 768 filenames listed with 'json' == 1 in 'cystods.csv' are not actually present in the
# 'images' folder on OSF. This is a minor inconsistency in the released dataset, so they are excluded here.
MISSING_IMAGES = {"0ba96ad9.png", "23a59431.png", "4336849b.png", "9699deab.png", "d412d37a.png"}


def _osf_list_all(url):
    import time
    import requests

    items = []
    expected_total = None
    while url:
        for attempt in range(5):
            r = requests.get(url)
            if r.status_code < 500:
                break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        payload = r.json()
        items.extend(payload["data"])
        if expected_total is None:
            expected_total = payload["links"]["meta"]["total"]
        url = payload["links"].get("next")
    return items, expected_total


def _get_osf_folder_download_urls(folder_id, attempts=3):
    url = f"https://api.osf.io/v2/nodes/{OSF_NODE_ID}/files/osfstorage/{folder_id}/?page[size]=100"
    for attempt in range(attempts):
        items, expected_total = _osf_list_all(url)
        if len(items) == expected_total:
            return {item["attributes"]["name"]: item["links"]["download"] for item in items}
        warn(
            f"Listing the OSF folder {folder_id} returned {len(items)} items, expected {expected_total}. "
            "Retrying." if attempt < attempts - 1 else "Giving up."
        )
    return {item["attributes"]["name"]: item["links"]["download"] for item in items}


def _download_with_retries(path, url, download, attempts=5):
    import time

    for attempt in range(attempts):
        try:
            util.download_source(path=path, url=url, download=download)
            return
        except Exception as e:
            if attempt == attempts - 1:
                raise
            warn(f"Download of {url} failed ({e}), retrying.")
            time.sleep(2 ** attempt)


def _get_segmented_filenames(csv_path):
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    filenames = [row["filename"] for row in rows if row.get("json") == "1"]
    return natsorted(fname for fname in filenames if fname not in MISSING_IMAGES)


def _rasterize_labels(segmentation_dir, filenames, label_dir):
    if os.path.exists(label_dir) and len(glob(os.path.join(label_dir, "*.tif"))) == len(filenames):
        return

    os.makedirs(label_dir, exist_ok=True)

    from skimage.draw import polygon as draw_polygon

    for fname in tqdm(filenames, desc="Rasterize the CystoDS annotations"):
        name = os.path.splitext(fname)[0]
        label_path = os.path.join(label_dir, f"{name}.tif")
        if os.path.exists(label_path):
            continue

        with open(os.path.join(segmentation_dir, f"{name}.json")) as f:
            annotation = json.load(f)

        image = imageio.imread(os.path.join(os.path.dirname(label_dir), "images", fname))
        shape = image.shape[:2]

        labels = np.zeros(shape, dtype="uint8")
        for shape_annotation in annotation["shapes"]:
            label_id = LABEL_MAP[shape_annotation["label"]]
            points = np.array(shape_annotation["points"], dtype=float)
            rows, columns = draw_polygon(points[:, 1], points[:, 0], shape=shape)
            labels[rows, columns] = label_id

        imageio.imwrite(label_path, labels, compression="zlib")


def get_cystods_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CystoDS dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder where the data is stored.
    """
    os.makedirs(path, exist_ok=True)

    csv_path = os.path.join(path, "cystods.csv")
    util.download_source(path=csv_path, url=CSV_URL, download=download, checksum=CSV_CHECKSUM)

    filenames = _get_segmented_filenames(csv_path)

    image_dir = os.path.join(path, "images")
    segmentation_dir = os.path.join(path, "segmentations")
    have_images = os.path.exists(image_dir) and len(glob(os.path.join(image_dir, "*.png"))) == len(filenames)
    have_segmentations = os.path.exists(segmentation_dir) and \
        len(glob(os.path.join(segmentation_dir, "*.json"))) == len(filenames)
    if have_images and have_segmentations:
        return path

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(segmentation_dir, exist_ok=True)

    image_urls = _get_osf_folder_download_urls(IMAGES_FOLDER_ID)
    segmentation_urls = _get_osf_folder_download_urls(SEGMENTATIONS_FOLDER_ID)

    for fname in tqdm(filenames, desc="Download the CystoDS images and segmentations"):
        name = os.path.splitext(fname)[0]
        segmentation_name = f"{name}.json"

        if fname not in image_urls or segmentation_name not in segmentation_urls:
            warn(f"'{fname}' is listed as segmented in 'cystods.csv' but missing from the OSF folders.")
            continue

        image_path = os.path.join(image_dir, fname)
        _download_with_retries(image_path, image_urls[fname], download)

        segmentation_path = os.path.join(segmentation_dir, segmentation_name)
        _download_with_retries(segmentation_path, segmentation_urls[segmentation_name], download)

    return path


def get_cystods_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the CystoDS data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_cystods_data(path, download)

    filenames = _get_segmented_filenames(os.path.join(data_dir, "cystods.csv"))
    # A few filenames may be missing an image or a segmentation on the OSF side (see 'get_cystods_data'),
    # so only keep the ones that were actually downloaded.
    filenames = [
        fname for fname in filenames
        if os.path.exists(os.path.join(data_dir, "images", fname))
        and os.path.exists(os.path.join(data_dir, "segmentations", f"{os.path.splitext(fname)[0]}.json"))
    ]

    label_dir = os.path.join(data_dir, "labels")
    _rasterize_labels(os.path.join(data_dir, "segmentations"), filenames, label_dir)

    raw_paths = natsorted(os.path.join(data_dir, "images", fname) for fname in filenames)
    label_paths = natsorted(glob(os.path.join(label_dir, "*.tif")))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(
        os.path.splitext(os.path.basename(raw_path))[0] == os.path.splitext(os.path.basename(label_path))[0]
        for raw_path, label_path in zip(raw_paths, label_paths)
    )

    return raw_paths, label_paths


def get_cystods_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the CystoDS dataset for bladder lesion and landmark segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cystods_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs,
    )


def get_cystods_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the CystoDS dataloader for bladder lesion and landmark segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cystods_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
