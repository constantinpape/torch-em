"""This dataset contains annotations for epithelium segmentation in colorectal tissue microarray
cores, stained with H&E, 13 immunohistochemistry (IHC) protein markers, and in situ hybridization
(ISH) for two microRNAs plus positive/negative controls.

The data is from the publication https://doi.org/10.1111/apm.70051 ("miR-143 and miR-145 in
Colorectal Cancer: A Digital Pathology Approach on Expressions and Protein Correlations"). It is
hosted on DataverseNO at https://doi.org/10.18710/DIGQGQ under a CC0-1.0 license. Please cite the
publication if you use this dataset in your research.

The dataset covers 100 patients, each with three normal mucosa and three cancer tissue microarray
cores per stain. Mask label values: 0 (background) and 1 (epithelium).

NOTE: Each stain is stored on DataverseNO as one multi-gigabyte zip archive of all its cores. To
avoid downloading a whole archive for the images actually requested, each raw/mask pair is fetched
as a single zip member via an HTTP range request.
"""

import os
import struct
import zlib
import zipfile
from glob import glob
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from tqdm import tqdm

import requests

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://dataverse.no/api/access/datafile"

URLS = {
    "HE": f"{BASE_URL}/233917",
    "ECAD": f"{BASE_URL}/233915",
    "VIMENTIN": f"{BASE_URL}/233933",
    "SMA": f"{BASE_URL}/233929",
    "Ki67": f"{BASE_URL}/233920",
    "SMAD3": f"{BASE_URL}/233934",
    "MACC1": f"{BASE_URL}/233918",
    "LASP1": f"{BASE_URL}/233921",
    "CD44": f"{BASE_URL}/233913",
    "NAIP": f"{BASE_URL}/233928",
    "KLF5": f"{BASE_URL}/233919",
    "FSCN1": f"{BASE_URL}/233916",
    "CTNND1": f"{BASE_URL}/233914",
    "KRAS": f"{BASE_URL}/233922",
    "miR-143": f"{BASE_URL}/234318",
    "miR-145": f"{BASE_URL}/233924",
    "U6": f"{BASE_URL}/233932",
    "Scr": f"{BASE_URL}/233930",
}

CHECKSUMS = {  # md5 of the full DataverseNO archives, kept for provenance
    "HE": "f82f503e821ee4c546ce3e5b82465cb7",
    "ECAD": "b83f719ebb1e46dbbfdfb17a784e467f",
    "VIMENTIN": "842bfcac56034fcb264d81c88e88455f",
    "SMA": "0debe0bf6613b6d5651c45fb7b84581d",
    "Ki67": "4060c46fa1334c3fe804346e26bf9225",
    "SMAD3": "f8a3cc5e556c833a384faa5914f66f08",
    "MACC1": "97b223362abcfaf3db010794300cf470",
    "LASP1": "d116d8cf53532a9729d25dbf5ee11ea1",
    "CD44": "c05e00ddb394f2d82cff3039c9fd667d",
    "NAIP": "b83bdbcd9211b94d1eba86353347e142",
    "KLF5": "fb48fb9dccb9de02b3c0e43f3f348410",
    "FSCN1": "1acfcc09f439eee9da58b26c836d980b",
    "CTNND1": "0866a21e48de86b7ff01daef02a7cea2",
    "KRAS": "63cee667a452c6f1e236b719d2eb78d3",
    "miR-143": "0bda5e92b2b4b3cf269879fb5f7e48ab",
    "miR-145": "76ee831cd078245ea54d78405b5dd8b4",
    "U6": "14918f009194e342843d284bd4a4d380",
    "Scr": "1e401fde4d37a721944cc23fc6b0a79c",
}

SPLIT_FOLDERS = {"cancer": "Cancer", "normal_mucosa": "Normal mucosa"}


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
        r = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        r.raise_for_status()
        data = r.content
        self.pos += len(data)
        return data

    def readable(self):
        return True

    def seekable(self):
        return True


def _remote_size(url):
    # A HEAD request fails on this host's presigned redirect target, which is only signed for GET.
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
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
    r = requests.get(url, headers={"Range": f"bytes={start}-{end}"})
    r.raise_for_status()
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


def _resolve_stains(stains):
    if stains is None:
        return list(URLS)
    if isinstance(stains, str):
        stains = [stains]
    invalid_stains = set(stains) - set(URLS)
    if invalid_stains:
        raise ValueError(f"Invalid stain choices: {sorted(invalid_stains)}. Choose from {sorted(URLS)}.")
    return stains


def _matches_split(member_name, split):
    return split is None or f"/{SPLIT_FOLDERS[split]}/" in member_name


def _matches_sample_ids(member_name, sample_ids):
    if sample_ids is None:
        return True
    stem = Path(member_name).stem
    return any(sample_id in stem for sample_id in sample_ids)


def get_crc_epithelium_data(
    path: Union[os.PathLike, str],
    stains: Optional[Union[str, List[str]]] = None,
    split: Optional[Literal["cancer", "normal_mucosa"]] = None,
    sample_ids: Optional[List[str]] = None,
    download: bool = False,
) -> str:
    """Download the CRC epithelium segmentation data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        stains: The stain(s) to download. By default all 18 stains are downloaded.
        split: The tissue split to restrict the data to. By default both splits are used.
        sample_ids: The core ids to restrict the data to, e.g. ['A001-4']. By default all cores are used.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the raw images and masks are stored.
    """
    stains = _resolve_stains(stains)
    path = str(path)
    os.makedirs(path, exist_ok=True)

    for stain in stains:
        url = URLS[stain]
        size, index = _list_zip_index(url)
        raw_members = sorted(
            name for name in index
            if name.lower().endswith(".jpg") and _matches_split(name, split) and _matches_sample_ids(name, sample_ids)
        )
        if not raw_members:
            raise RuntimeError(f"No members of stain '{stain}' match the requested 'split' / 'sample_ids'.")

        for raw_name in tqdm(raw_members, desc=f"Fetch {stain} cores"):
            mask_name = raw_name[:-len(".jpg")] + ".png"
            if mask_name not in index:
                raise RuntimeError(f"Missing mask '{mask_name}' for raw image '{raw_name}' in stain '{stain}'.")

            raw_path = os.path.join(path, raw_name)
            mask_path = os.path.join(path, mask_name)
            if os.path.exists(raw_path) and os.path.exists(mask_path):
                continue
            if not download:
                raise RuntimeError(f"Data for stain '{stain}' is not found and download is set to False.")

            _fetch_zip_member(url, size, index[raw_name], raw_path)
            _fetch_zip_member(url, size, index[mask_name], mask_path)

    return path


def get_crc_epithelium_paths(
    path: Union[os.PathLike, str],
    stains: Optional[Union[str, List[str]]] = None,
    split: Optional[Literal["cancer", "normal_mucosa"]] = None,
    sample_ids: Optional[List[str]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CRC epithelium segmentation images and masks.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        stains: The stain(s) to use. By default all 18 stains are used.
        split: The tissue split to restrict the data to. By default both splits are used.
        sample_ids: The core ids to restrict the data to, e.g. ['A001-4']. By default all cores are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    stains = _resolve_stains(stains)
    data_dir = get_crc_epithelium_data(path, stains, split, sample_ids, download)

    raw_paths, label_paths = [], []
    for stain in stains:
        stain_raw_paths = sorted(glob(os.path.join(data_dir, stain, "**", "*.jpg"), recursive=True))
        for raw_path in stain_raw_paths:
            member_name = os.path.relpath(raw_path, data_dir)
            if not (_matches_split(member_name, split) and _matches_sample_ids(member_name, sample_ids)):
                continue
            mask_path = os.path.splitext(raw_path)[0] + ".png"
            if not os.path.exists(mask_path):
                raise RuntimeError(f"Missing mask for raw image '{raw_path}'.")
            raw_paths.append(raw_path)
            label_paths.append(mask_path)

    if not raw_paths:
        raise RuntimeError("Could not find any images and masks for the requested settings.")

    return raw_paths, label_paths


def get_crc_epithelium_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    stains: Optional[Union[str, List[str]]] = None,
    split: Optional[Literal["cancer", "normal_mucosa"]] = None,
    sample_ids: Optional[List[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the CRC epithelium segmentation dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        stains: The stain(s) to use. By default all 18 stains are used.
        split: The tissue split to restrict the data to. By default both splits are used.
        sample_ids: The core ids to restrict the data to, e.g. ['A001-4']. By default all cores are used.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_crc_epithelium_paths(path, stains, split, sample_ids, download)

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


def get_crc_epithelium_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    stains: Optional[Union[str, List[str]]] = None,
    split: Optional[Literal["cancer", "normal_mucosa"]] = None,
    sample_ids: Optional[List[str]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the CRC epithelium segmentation dataloader.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        stains: The stain(s) to use. By default all 18 stains are used.
        split: The tissue split to restrict the data to. By default both splits are used.
        sample_ids: The core ids to restrict the data to, e.g. ['A001-4']. By default all cores are used.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_crc_epithelium_dataset(
        path, patch_shape, stains, split, sample_ids, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
