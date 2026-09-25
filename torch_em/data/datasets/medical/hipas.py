"""The HiPaS dataset contains annotations for pulmonary artery and vein segmentation in non-contrast chest CT.

The dataset consists of 250 non-contrast CT scans (512 x 512 in-plane, about 0.5 - 0.9 mm in-plane and 1 mm
slice spacing), each with a binary artery and a binary vein mask. `get_hipas_data` converts every case to a single
hdf5 file with the slice axis first (the keys are 'raw' and 'labels'), where the labels are a semantic volume
following `LABEL_IDS`: 1 = pulmonary artery, 2 = pulmonary vein. The artery and vein masks overlap in about 1 - 2 %
of their voxels, these voxels are labeled as vein.

NOTE: This is not the same data as the already-integrated `torch_em.data.datasets.medical.airrc`, which provides
artery, vein and airway masks on resampled crops of the LUNA16 CT scans. HiPaS is a separate collection with its
own CT scans (the AirRC publication uses it as an independent external benchmark).

The CT scans are stored in a single ~24 GB zip archive. To support downloading only a subset of the cases, the
individual scans are read from the archive with HTTP range requests instead of downloading it as a whole.

The data is located at https://doi.org/10.5281/zenodo.14879605 (the example data of the publication, released under
an MIT license).

This dataset is from the publication https://doi.org/10.1038/s41467-025-56505-6.
Please cite it if you use this dataset for your research.
"""

import io
import os
import json
import uuid
import zlib
import struct
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Optional
from concurrent import futures

import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL_BASE = "https://zenodo.org/api/records/14879605/files"
ANNOTATION_CHECKSUM = "8badcebd8c82d5ac0c1ea38d08d482efd37c78712009e7839f8339eee11f82ce"

LABEL_IDS = {"artery": 1, "vein": 2}
"""Mapping from the name of a vessel type to its label id in the converted label volumes."""


def _read_zip_entries(url):
    import requests

    size = int(requests.head(url, allow_redirects=True).headers["Content-Length"])
    tail = requests.get(url, headers={"Range": f"bytes={size - 65536}-{size - 1}"}).content
    eocd = tail.rfind(b"PK\x05\x06")
    n_entries, cd_size, cd_offset = struct.unpack("<HII", tail[eocd + 10:eocd + 20])
    if n_entries == 0xFFFF or cd_offset == 0xFFFFFFFF:
        locator = tail.rfind(b"PK\x06\x07")
        zip64_offset = struct.unpack("<Q", tail[locator + 8:locator + 16])[0]
        zip64 = requests.get(url, headers={"Range": f"bytes={zip64_offset}-{zip64_offset + 55}"}).content
        n_entries, cd_size, cd_offset = struct.unpack("<QQQ", zip64[32:56])
    directory = requests.get(url, headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"}).content

    entries, pos = {}, 0
    for _ in range(n_entries):
        fields = struct.unpack("<IHHHHHHIIIHHHHHII", directory[pos:pos + 46])
        method, csize, usize = fields[4], fields[8], fields[9]
        name_len, extra_len, comment_len = fields[10], fields[11], fields[12]
        header_offset = fields[16]
        name = directory[pos + 46:pos + 46 + name_len].decode()
        extra = directory[pos + 46 + name_len:pos + 46 + name_len + extra_len]
        offset = 0
        while offset < len(extra):
            tag, field_size = struct.unpack("<HH", extra[offset:offset + 4])
            if tag == 1:
                field, field_pos = extra[offset + 4:offset + 4 + field_size], 0
                if usize == 0xFFFFFFFF:
                    usize, field_pos = struct.unpack("<Q", field[field_pos:field_pos + 8])[0], field_pos + 8
                if csize == 0xFFFFFFFF:
                    csize, field_pos = struct.unpack("<Q", field[field_pos:field_pos + 8])[0], field_pos + 8
                if header_offset == 0xFFFFFFFF:
                    header_offset = struct.unpack("<Q", field[field_pos:field_pos + 8])[0]
            offset += 4 + field_size
        entries[name] = {"method": method, "compressed_size": csize, "header_offset": header_offset}
        pos += 46 + name_len + extra_len + comment_len

    return entries


def _get_zip_entries(path, url):
    cache_path = os.path.join(path, "ct_scan_entries.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    entries = _read_zip_entries(url)
    tmp_path = f"{cache_path}.{uuid.uuid4().hex}.incomplete"
    with open(tmp_path, "w") as f:
        json.dump(entries, f)
    os.replace(tmp_path, cache_path)
    return entries


def _read_zip_member(url, entry):
    import requests

    offset = entry["header_offset"]
    header = requests.get(url, headers={"Range": f"bytes={offset}-{offset + 29}"}).content
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    start = offset + 30 + name_len + extra_len
    response = requests.get(url, headers={"Range": f"bytes={start}-{start + entry['compressed_size'] - 1}"})
    response.raise_for_status()
    return zlib.decompress(response.content, -15) if entry["method"] == 8 else response.content


def _convert_case(case_id, path, entries, ct_url):
    import h5py

    out_path = os.path.join(path, "preprocessed", f"{case_id}.h5")
    if os.path.exists(out_path):
        return

    ct = np.load(io.BytesIO(_read_zip_member(ct_url, entries[f"ct_scan/{case_id}.npz"])))["data"]
    labels = np.zeros(ct.shape, dtype="uint8")
    for name, label_id in LABEL_IDS.items():
        mask = np.load(os.path.join(path, "annotation", name, f"{case_id}.npz"))["data"]
        assert mask.shape == ct.shape, f"{case_id}: {mask.shape} != {ct.shape}"
        labels[mask > 0] = label_id

    ct, labels = ct.transpose(2, 0, 1), labels.transpose(2, 0, 1)
    chunks = (min(16, ct.shape[0]), 128, 128)
    tmp_path = f"{out_path}.{uuid.uuid4().hex}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw", data=ct, chunks=chunks, compression="gzip")
        f.create_dataset("labels", data=labels, chunks=chunks, compression="gzip")
    os.replace(tmp_path, out_path)


def get_hipas_data(
    path: Union[os.PathLike, str], n_cases: Optional[int] = None, n_workers: int = 4, download: bool = False
) -> str:
    """Download the HiPaS dataset and convert the cases to hdf5 files.

    NOTE: The full collection is about 24 GB. Use `n_cases` to only download a subset for a quick start.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        n_cases: The number of cases to download, sorted by case id. By default all 250 cases are downloaded.
        n_workers: The number of parallel download and conversion workers.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the converted hdf5 files.
    """
    os.makedirs(path, exist_ok=True)

    annotation_dir = os.path.join(path, "annotation")
    if not os.path.exists(annotation_dir):
        zip_path = os.path.join(path, "annotation.zip")
        util.download_source(
            path=zip_path, url=f"{URL_BASE}/annotation.zip/content", download=download, checksum=ANNOTATION_CHECKSUM,
        )
        util.unzip(zip_path=zip_path, dst=path, remove=False)

    case_ids = natsorted(f[:-4] for f in os.listdir(os.path.join(annotation_dir, "artery")) if f.endswith(".npz"))
    if n_cases is not None:
        case_ids = case_ids[:n_cases]

    preprocessed_dir = os.path.join(path, "preprocessed")
    missing = [c for c in case_ids if not os.path.exists(os.path.join(preprocessed_dir, f"{c}.h5"))]
    if not missing:
        return preprocessed_dir
    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    os.makedirs(preprocessed_dir, exist_ok=True)
    ct_url = f"{URL_BASE}/ct_scan.zip/content"
    entries = _get_zip_entries(path, ct_url)
    with futures.ThreadPoolExecutor(n_workers) as pool:
        tasks = [pool.submit(_convert_case, case_id, path, entries, ct_url) for case_id in missing]
        for task in tqdm(futures.as_completed(tasks), total=len(tasks), desc="Download and convert HiPaS"):
            task.result()

    return preprocessed_dir


def get_hipas_paths(
    path: Union[os.PathLike, str], n_cases: Optional[int] = None, download: bool = False,
) -> List[str]:
    """Get paths to the HiPaS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        n_cases: The number of cases to use, sorted by case id. By default all 250 cases are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    data_dir = get_hipas_data(path, n_cases, download=download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    if n_cases is not None:
        volume_paths = volume_paths[:n_cases]
    return volume_paths


def get_hipas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    n_cases: Optional[int] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HiPaS dataset for pulmonary artery and vein segmentation in non-contrast chest CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        n_cases: The number of cases to use, sorted by case id. By default all 250 cases are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_hipas_paths(path, n_cases, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        is_seg_dataset=True,
        patch_shape=patch_shape,
        ndim=3,
        **kwargs
    )


def get_hipas_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    n_cases: Optional[int] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the HiPaS dataloader for pulmonary artery and vein segmentation in non-contrast chest CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        n_cases: The number of cases to use, sorted by case id. By default all 250 cases are used.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hipas_dataset(path, patch_shape, n_cases, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
