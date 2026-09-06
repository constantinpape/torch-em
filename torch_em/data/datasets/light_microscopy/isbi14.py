"""The ISBI14 dataset contains cervical cytology images with nucleus and cytoplasm
instance segmentation annotations, released for the Overlapping Cervical Cytology Image
Segmentation Challenge at ISBI 2014.

The dataset has two image sources:
- 'synthetic': 945 synthetic Pap-smear images (45 train, 900 test) composed from the real
  images below, with per-cell nucleus and cytoplasm instance masks.
- 'real': 16 real extended-depth-of-field (EDF) Pap-smear images with nucleus instance masks.
  No cytoplasm annotation is available for the real images.

NOTE: 4 of the 16 real images (EDF000-EDF003) ship ground truth as hundreds of few-pixel
fragments instead of per-cell nucleus masks. This loader drops them and only exposes the
remaining 12 real images.

NOTE: No data license is published on the challenge site, its linked pages, or inside the
released archives.

The dataset is located at https://cs.adelaide.edu.au/~carneiro/isbi14_challenge/dataset.html.
This dataset is from the publications https://doi.org/10.1109/JBHI.2016.2601609 and
https://doi.org/10.1109/TIP.2015.2389619. Please cite them if you use this dataset in your research.
"""

import os
import ssl
from glob import glob
from natsort import natsorted
from typing import List, Literal, Tuple, Union

import numpy as np
import requests
import h5py
import scipy.io as sio
import imageio.v3 as imageio
from scipy import ndimage
from requests.adapters import HTTPAdapter
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://cs.adelaide.edu.au/~carneiro/isbi14_challenge/Dataset.zip"
CHECKSUM = "618e386cd87722ad0bca5ebe8570fc7d96ba4cf011eb8b018f865c4e7d8328f0"

# The challenge server only supports legacy TLS renegotiation, which OpenSSL 3 disables by default.
SSL_OP_LEGACY_SERVER_CONNECT = 0x4

# These real images ship ground truth as hundreds of few-pixel fragments, not usable nucleus masks.
BROKEN_REAL_IMAGES = ("EDF000", "EDF001", "EDF002", "EDF003")

IMAGE_SOURCES = ("synthetic", "real")
LABEL_CHOICES = ("nucleus", "cytoplasm")


class _LegacyRenegotiationAdapter(HTTPAdapter):
    """Allow the TLS renegotiation that the challenge server still requires."""

    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.options |= SSL_OP_LEGACY_SERVER_CONNECT
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)


def _download_zip(zip_path: str, download: bool) -> None:
    if os.path.exists(zip_path):
        return
    if not download:
        raise RuntimeError(f"Cannot find the data at {zip_path}, but download was set to False")

    session = requests.Session()
    session.mount("https://cs.adelaide.edu.au", _LegacyRenegotiationAdapter())

    tmp_path = f"{zip_path}.incomplete"
    with session.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        file_size = int(r.headers.get("Content-Length", 0))
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc=f"Download {URL}") as r_raw, open(tmp_path, "wb") as f:
            for chunk in iter(lambda: r_raw.read(1024 * 1024), b""):
                f.write(chunk)

    this_checksum = util.get_checksum(tmp_path)
    if this_checksum != CHECKSUM:
        raise RuntimeError(f"The checksum of the download does not match. Expected: {CHECKSUM}, got: {this_checksum}")
    os.replace(tmp_path, zip_path)


def get_isbi14_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ISBI14 dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data directory.
    """
    data_dir = os.path.join(path, "Dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "Dataset.zip")
    _download_zip(zip_path, download)
    util.unzip(zip_path, path)

    return data_dir


def _load_mat_array(f_or_dict, key, index, is_h5):
    """Load one entry of a MATLAB cell array, from either a scipy or an h5py backed file."""
    if is_h5:
        return np.array(f_or_dict[f_or_dict[key][0, index]][()]).T
    return f_or_dict[key][index, 0]


def _load_cytoplasm_entry(f_or_dict, key, index, is_h5):
    """Load the per-cell cytoplasm masks of one image, from either a scipy or an h5py backed file."""
    if is_h5:
        refs = np.array(f_or_dict[f_or_dict[key][0, index]][()]).ravel()
        return [np.array(f_or_dict[ref][()]).T for ref in refs]
    entry = f_or_dict[key][index, 0]
    return [entry[j, 0] for j in range(entry.shape[0])]


def _cytoplasm_to_labels(cytoplasm_masks: List[np.ndarray]) -> np.ndarray:
    """Merge per-cell cytoplasm masks into one label image, larger cells first so a smaller,
    more visible cell on top keeps its own label where cytoplasms overlap."""
    ordered = sorted(range(len(cytoplasm_masks)), key=lambda i: int(cytoplasm_masks[i].sum()), reverse=True)
    labels = np.zeros_like(cytoplasm_masks[0], dtype="uint16")
    for instance_id, i in enumerate(ordered, start=1):
        labels[cytoplasm_masks[i] > 0] = instance_id
    return labels


def _prepare_synthetic_split(data_dir: str, split: Literal["train", "test"]) -> Tuple[str, str, str]:
    raw_dir = os.path.join(data_dir, "synthetic_preprocessed", split, "raw")
    nucleus_dir = os.path.join(data_dir, "synthetic_preprocessed", split, "labels_nucleus")
    cytoplasm_dir = os.path.join(data_dir, "synthetic_preprocessed", split, "labels_cytoplasm")

    raw_mat = os.path.join(data_dir, "Synthetic", f"{split}set.mat")
    gt_mat = os.path.join(data_dir, "Synthetic", f"{split}set_GT.mat")

    try:
        gt = sio.loadmat(gt_mat)
        gt_is_h5 = False
    except NotImplementedError:
        gt = h5py.File(gt_mat, "r")
        gt_is_h5 = True

    n_images = gt[f"{split}_Nuclei"].shape[0 if not gt_is_h5 else 1]

    is_cached = all(
        os.path.exists(d) and len(glob(os.path.join(d, "*.tif"))) == n_images
        for d in (raw_dir, nucleus_dir, cytoplasm_dir)
    )
    if is_cached:
        if gt_is_h5:
            gt.close()
        return raw_dir, nucleus_dir, cytoplasm_dir

    for d in (raw_dir, nucleus_dir, cytoplasm_dir):
        os.makedirs(d, exist_ok=True)

    with h5py.File(raw_mat, "r") as raw_f:
        for i in tqdm(range(n_images), desc=f"Preprocess ISBI14 synthetic {split} images"):
            name = f"{i:04d}.tif"

            raw = _load_mat_array(raw_f, f"{split}set", i, is_h5=True)
            imageio.imwrite(os.path.join(raw_dir, name), raw)

            nucleus_mask = _load_mat_array(gt, f"{split}_Nuclei", i, gt_is_h5)
            nucleus_labels, _ = ndimage.label(nucleus_mask)
            imageio.imwrite(os.path.join(nucleus_dir, name), nucleus_labels.astype("uint16"))

            cytoplasm_masks = _load_cytoplasm_entry(gt, f"{split}_Cytoplasm", i, gt_is_h5)
            cytoplasm_labels = _cytoplasm_to_labels(cytoplasm_masks)
            imageio.imwrite(os.path.join(cytoplasm_dir, name), cytoplasm_labels)

    if gt_is_h5:
        gt.close()

    return raw_dir, nucleus_dir, cytoplasm_dir


def _prepare_real(data_dir: str) -> Tuple[str, str]:
    raw_dir = os.path.join(data_dir, "real_preprocessed", "raw")
    label_dir = os.path.join(data_dir, "real_preprocessed", "labels_nucleus")

    image_paths = [
        p for p in natsorted(glob(os.path.join(data_dir, "EDF", "*.png")))
        if not os.path.basename(p).endswith("_GT.png")
        and os.path.splitext(os.path.basename(p))[0] not in BROKEN_REAL_IMAGES
    ]
    is_cached = all(
        os.path.exists(d) and len(glob(os.path.join(d, "*.tif"))) == len(image_paths)
        for d in (raw_dir, label_dir)
    )
    if is_cached:
        return raw_dir, label_dir

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Preprocess ISBI14 real images"):
        name = os.path.splitext(os.path.basename(image_path))[0]
        gt_path = os.path.join(data_dir, "EDF", f"{name}_GT.png")

        raw = imageio.imread(image_path)
        mask = imageio.imread(gt_path) > 0
        labels, _ = ndimage.label(mask)

        imageio.imwrite(os.path.join(raw_dir, f"{name}.tif"), raw)
        imageio.imwrite(os.path.join(label_dir, f"{name}.tif"), labels.astype("uint16"))

    return raw_dir, label_dir


def get_isbi14_paths(
    path: Union[os.PathLike, str],
    image_source: Literal["synthetic", "real"] = "synthetic",
    label_choice: Literal["nucleus", "cytoplasm"] = "nucleus",
    split: Literal["train", "test"] = "train",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the ISBI14 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        image_source: The image source. Either 'synthetic' or 'real'.
        label_choice: The segmentation target. Either 'nucleus' or 'cytoplasm'.
            No cytoplasm annotation exists for the real images.
        split: The data split. Only used for the synthetic images.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if image_source not in IMAGE_SOURCES:
        raise ValueError(f"'{image_source}' is not a valid image source. Choose from {list(IMAGE_SOURCES)}.")
    if label_choice not in LABEL_CHOICES:
        raise ValueError(f"'{label_choice}' is not a valid label choice. Choose from {list(LABEL_CHOICES)}.")
    if image_source == "real" and label_choice == "cytoplasm":
        raise ValueError("No cytoplasm annotation exists for the real images. Use label_choice='nucleus'.")

    data_dir = get_isbi14_data(path, download)

    if image_source == "real":
        raw_dir, label_dir = _prepare_real(data_dir)
    else:
        raw_dir, nucleus_dir, cytoplasm_dir = _prepare_synthetic_split(data_dir, split)
        label_dir = nucleus_dir if label_choice == "nucleus" else cytoplasm_dir

    raw_paths = natsorted(glob(os.path.join(raw_dir, "*.tif")))
    label_paths = natsorted(glob(os.path.join(label_dir, "*.tif")))

    return raw_paths, label_paths


def get_isbi14_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    image_source: Literal["synthetic", "real"] = "synthetic",
    label_choice: Literal["nucleus", "cytoplasm"] = "nucleus",
    split: Literal["train", "test"] = "train",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the ISBI14 dataset for cervical cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        image_source: The image source. Either 'synthetic' or 'real'.
        label_choice: The segmentation target. Either 'nucleus' or 'cytoplasm'.
            No cytoplasm annotation exists for the real images.
        split: The data split. Only used for the synthetic images.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_isbi14_paths(path, image_source, label_choice, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_isbi14_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    image_source: Literal["synthetic", "real"] = "synthetic",
    label_choice: Literal["nucleus", "cytoplasm"] = "nucleus",
    split: Literal["train", "test"] = "train",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the ISBI14 dataloader for cervical cell instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        image_source: The image source. Either 'synthetic' or 'real'.
        label_choice: The segmentation target. Either 'nucleus' or 'cytoplasm'.
            No cytoplasm annotation exists for the real images.
        split: The data split. Only used for the synthetic images.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_isbi14_dataset(path, patch_shape, image_source, label_choice, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
