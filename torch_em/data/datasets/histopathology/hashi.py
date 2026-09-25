"""This dataset contains annotations for invasive breast cancer region segmentation
in H&E-stained whole-slide histopathology tiles from four institutions.

The data is from the publication https://doi.org/10.1371/journal.pone.0196828
("High-throughput adaptive sampling for whole-slide histopathology image analysis
(HASHI) via convolutional neural networks: application to invasive breast cancer
detection"). It is hosted on Zenodo at https://zenodo.org/records/4993672 under a
CC0-1.0 license. Please cite the publication if you use this dataset in your research.

The dataset covers four cohorts: HUP and UHCMC/CWRU (used for training in the
publication), and CINJ and TCGA (held out for testing). Three pathologists manually
delineated invasive cancer regions at 2x magnification. Mask label values: 0
(background) and 1 (invasive cancer region).

NOTE: Each cohort's images and masks are stored in separate multi-slide zip archives
on Zenodo. To avoid downloading a whole archive for a subset of its slides, each
raw/mask pair is fetched as a single zip member via an HTTP range request. Raw tiles
and their masks can differ by a 1px margin on each axis; each pair is cropped to their
shared shape on first download.
"""

import os
import time
import struct
import zlib
import zipfile
from glob import glob
from natsort import natsorted
from typing import List, Literal, Optional, Tuple, Union

from tqdm import tqdm

import imageio.v3 as imageio

import requests

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


RECORD_URL = "https://zenodo.org/api/records/4993672/files"
N_RETRIES = 5

COHORTS = {
    "hup": {"imgs": ["HUP_imgs_idx5_Part1.zip", "HUP_imgs_idx5_Part2.zip"], "masks": "HUP_masks.zip"},
    "cwru": {"imgs": ["CWRU_imgs_idx8.zip"], "masks": "CWRU_masks.zip"},
    "cinj": {"imgs": ["CINJ_imgs_idx5.zip"], "masks": "CINJ_masks_HG.zip"},
    "tcga": {"imgs": ["TCGA_imgs_idx5.zip"], "masks": "TCGA_masks.zip"},
}

CHECKSUMS = {  # md5 of the full Zenodo archives, kept for provenance
    "HUP_imgs_idx5_Part1.zip": "c1bdaeb3c5bd2cd657b0081b9f52e3fe",
    "HUP_imgs_idx5_Part2.zip": "3127ee09f1752b76edd8d16383c52d30",
    "HUP_masks.zip": "6785a4cc69eae45217eb3e39843e3465",
    "CWRU_imgs_idx8.zip": "b1d13b2ecec81c0efe877bb25f45ece1",
    "CWRU_masks.zip": "a83e46a69d99c8e88ba5beef3be654a9",
    "CINJ_imgs_idx5.zip": "76dd6dc6f2e78bacdec427d0bd0d8740",
    "CINJ_masks_HG.zip": "65ab52631acaa40085972d1946ed1924",
    "TCGA_imgs_idx5.zip": "cc62330b4a2421219bd38f958e901c13",
    "TCGA_masks.zip": "a48c5c04c93d01831dc4a89a4dba609a",
}

# HUP and CINJ raw tiles are named '{id}_idx5.png', with masks named differently.
# CWRU and TCGA raw tiles and masks share the exact same filename.
MASK_SUFFIX = {"hup": "_annotation_mask.png", "cinj": ".png"}
SPLITS = {"train": ["hup", "cwru"], "test": ["cinj", "tcga"]}


def _get_with_retries(url, **kwargs):
    # The Zenodo API gateway occasionally times out under range requests; retry with backoff.
    for attempt in range(N_RETRIES):
        try:
            r = requests.get(url, timeout=60, **kwargs)
            r.raise_for_status()
            return r
        except (requests.exceptions.RequestException,) as error:
            if attempt == N_RETRIES - 1:
                raise error
            time.sleep(2 ** attempt)


class _RemoteZipReader:
    """Seekable file-like object over a remote zip, for reading its (small) structural data."""

    def __init__(self, url, size):
        self.url = url
        self.size = size
        self.pos = 0

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        end = self.size - 1 if n is None or n < 0 else min(self.pos + n, self.size) - 1
        if end < self.pos:
            return b""
        r = _get_with_retries(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = r.content
        self.pos += len(data)
        return data

    def readable(self):
        return True

    def seekable(self):
        return True


def _remote_size(url):
    # A HEAD request fails on this host's presigned redirect target, which is only signed for GET.
    r = _get_with_retries(url, stream=True, allow_redirects=True)
    return int(r.headers["Content-Length"])


def _list_zip_index(url):
    """List a remote zip's central directory once, so individual members can be fetched by offset."""
    size = _remote_size(url)
    zf = zipfile.ZipFile(_RemoteZipReader(url, size))
    return size, {info.filename: info for info in zf.infolist()}


def _fetch_zip_member(url, size, info, dst_path):
    if os.path.exists(dst_path):
        return

    # Fetch the local file header (with headroom for its variable-length fields) and the compressed
    # payload in one range request. This dataset's members carry short names and no extra fields, so
    # 256 bytes of headroom is always enough.
    start = info.header_offset
    end = start + 256 + info.compress_size - 1
    r = _get_with_retries(url, headers={"Range": f"bytes={start}-{end}"})
    buf = r.content

    fn_len, extra_len = struct.unpack("<HH", buf[26:30])
    data_start = 30 + fn_len + extra_len
    compressed = buf[data_start:data_start + info.compress_size]
    if len(compressed) != info.compress_size:
        raise RuntimeError(f"Incomplete read for zip member '{info.filename}'.")

    data = compressed if info.compress_type == zipfile.ZIP_STORED else zlib.decompressobj(-15).decompress(compressed)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = dst_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, dst_path)


def _align_pair(img_path, mask_path):
    """Crop a raw tile and its mask to their shared shape, in case they differ by a small margin."""
    image = imageio.imread(img_path)
    mask = imageio.imread(mask_path)
    height, width = min(image.shape[0], mask.shape[0]), min(image.shape[1], mask.shape[1])
    if image.shape[:2] != (height, width):
        imageio.imwrite(img_path, image[:height, :width])
    if mask.shape[:2] != (height, width):
        imageio.imwrite(mask_path, mask[:height, :width])


def _image_key(cohort, basename):
    return basename[:-len("_idx5.png")] if cohort in MASK_SUFFIX else basename


def _mask_key(cohort, basename):
    suffix = MASK_SUFFIX.get(cohort)
    return basename[:-len(suffix)] if suffix else basename


def _resolve_cohorts(cohorts):
    if cohorts is None:
        return list(COHORTS)
    if isinstance(cohorts, str):
        cohorts = [cohorts]
    invalid = set(cohorts) - set(COHORTS)
    if invalid:
        raise ValueError(f"Invalid cohort choices: {sorted(invalid)}. Choose from {sorted(COHORTS)}.")
    return cohorts


def get_hashi_data(
    path: Union[os.PathLike, str],
    cohorts: Optional[Union[str, List[str]]] = None,
    sample_ids: Optional[List[str]] = None,
    download: bool = False,
) -> str:
    """Download the HASHI invasive breast cancer segmentation data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        cohorts: The cohort(s) to download. By default all four cohorts are downloaded.
        sample_ids: The tile ids to restrict the data to. By default all tiles are used.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the raw images and masks are stored.
    """
    cohorts = _resolve_cohorts(cohorts)
    path = str(path)
    os.makedirs(path, exist_ok=True)

    for cohort in cohorts:
        masks_url = f"{RECORD_URL}/{COHORTS[cohort]['masks']}/content"
        masks_size, masks_index = _list_zip_index(masks_url)
        mask_by_key = {
            _mask_key(cohort, name.rsplit("/", 1)[-1]): name
            for name in masks_index if name.lower().endswith(".png")
        }

        for imgs_zip in COHORTS[cohort]["imgs"]:
            imgs_url = f"{RECORD_URL}/{imgs_zip}/content"
            imgs_size, imgs_index = _list_zip_index(imgs_url)
            img_members = sorted(name for name in imgs_index if name.lower().endswith(".png"))

            for img_name in tqdm(img_members, desc=f"Fetch {cohort} tiles ({imgs_zip})"):
                basename = img_name.rsplit("/", 1)[-1]
                key = _image_key(cohort, basename)
                if sample_ids is not None and key not in sample_ids:
                    continue
                if key not in mask_by_key:
                    raise RuntimeError(f"Missing mask for raw image '{img_name}' in cohort '{cohort}'.")
                mask_name = mask_by_key[key]

                img_path = os.path.join(path, cohort, "images", basename)
                mask_path = os.path.join(path, cohort, "masks", mask_name.rsplit("/", 1)[-1])
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    continue
                if not download:
                    raise RuntimeError(f"Data for cohort '{cohort}' is not found and download is set to False.")

                _fetch_zip_member(imgs_url, imgs_size, imgs_index[img_name], img_path)
                _fetch_zip_member(masks_url, masks_size, masks_index[mask_name], mask_path)
                _align_pair(img_path, mask_path)

    return path


def get_hashi_paths(
    path: Union[os.PathLike, str],
    cohorts: Optional[Union[str, List[str]]] = None,
    split: Optional[Literal["train", "test"]] = None,
    sample_ids: Optional[List[str]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the HASHI invasive breast cancer segmentation images and masks.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        cohorts: The cohort(s) to use. By default all four cohorts are used.
        split: The documented train ('hup', 'cwru') / test ('cinj', 'tcga') split. By default both are used.
        sample_ids: The tile ids to restrict the data to. By default all tiles are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    cohorts = _resolve_cohorts(cohorts)
    if split is not None:
        cohorts = [cohort for cohort in cohorts if cohort in SPLITS[split]]

    data_dir = get_hashi_data(path, cohorts, sample_ids, download)

    raw_paths, label_paths = [], []
    for cohort in cohorts:
        img_paths = natsorted(glob(os.path.join(data_dir, cohort, "images", "*.png")))
        mask_dir = os.path.join(data_dir, cohort, "masks")
        mask_by_key = {_mask_key(cohort, name): name for name in os.listdir(mask_dir)}

        for img_path in img_paths:
            basename = os.path.basename(img_path)
            key = _image_key(cohort, basename)
            if sample_ids is not None and key not in sample_ids:
                continue
            if key not in mask_by_key:
                raise RuntimeError(f"Missing mask for raw image '{img_path}' in cohort '{cohort}'.")
            raw_paths.append(img_path)
            label_paths.append(os.path.join(mask_dir, mask_by_key[key]))

    if not raw_paths:
        raise RuntimeError("Could not find any images and masks for the requested settings.")

    return raw_paths, label_paths


def get_hashi_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    cohorts: Optional[Union[str, List[str]]] = None,
    split: Optional[Literal["train", "test"]] = None,
    sample_ids: Optional[List[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the HASHI dataset for invasive breast cancer region segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        cohorts: The cohort(s) to use. By default all four cohorts are used.
        split: The documented train ('hup', 'cwru') / test ('cinj', 'tcga') split. By default both are used.
        sample_ids: The tile ids to restrict the data to. By default all tiles are used.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_hashi_paths(path, cohorts, split, sample_ids, download)

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
        ndim=2,
        with_channels=True,
        **kwargs,
    )


def get_hashi_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    cohorts: Optional[Union[str, List[str]]] = None,
    split: Optional[Literal["train", "test"]] = None,
    sample_ids: Optional[List[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the HASHI dataloader for invasive breast cancer region segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        cohorts: The cohort(s) to use. By default all four cohorts are used.
        split: The documented train ('hup', 'cwru') / test ('cinj', 'tcga') split. By default both are used.
        sample_ids: The tile ids to restrict the data to. By default all tiles are used.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hashi_dataset(
        path, patch_shape, cohorts, split, sample_ids, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
