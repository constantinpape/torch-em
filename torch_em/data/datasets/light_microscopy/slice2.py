"""The SLICE-2 dataset contains annotations for 3D nucleus instance segmentation in light-sheet fluorescence
microscopy volumes of Tribolium castaneum embryos.

SLICE-2 (Second Systematic Live Imaging Collection of Embryogenesis) is a collection of sixteen isotropic 3D
live imaging datasets of gastrulation and early germband elongation of the red flour beetle, imaged with a
histone-labeled (H2A/H2B) transgenic line. The nuclei of selected time points are segmented, so that this module
provides 15 labeled volumes from 14 of the datasets (`DATASETS` lists the labeled time points of each). The
raw data are the deconvolved z-stacks ('(D1)-ZStacks-Decon-ZS', about 600 x 1100 x 600 voxels, uint16), and the labels
are nucleus instance ids on the same grid. The segmentation is stored by the authors in a different axis order
(1000 x 600 x 600, flipped along one axis), which `get_slice2_data` converts. It was checked against the raw
volumes, where the segmented voxels are 4 - 8 times brighter than the background.

The published segmentation contains objects with the reserved id 65535 (about 6% of the segmented voxels, many small
fragments). These voxels are set to background and the remaining ids are made consecutive. The segmentation of
DS0001, DS0003 and further time points is only distributed as colored 'CH(TM)' RGB stacks, which do not match the
raw grid of the instance stacks. They are not used by this module.

NOTE: Every volume takes about 0.55 GB (raw) plus the labels to download. The raw data are stored in ~35 GB zip
archives on Zenodo (one per dataset), which are never downloaded as a whole: the members are read with HTTP range
requests and checked against the CRC32 of the archive. Use `datasets` to only download a subset of the datasets.

The data is located at https://doi.org/10.5281/zenodo.18015798 (segmentations) and in one Zenodo record per raw
archive (see `DATASETS`), released under a CC-BY-4.0 license.

This dataset is from the publication Kraemer et al. (2026), see the Zenodo records for the details.
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
from typing import Union, Tuple, Optional, List, Sequence
from concurrent import futures

import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL_BASE = "https://zenodo.org/api/records"
SEGMENTATION_RECORD = 20391579

DATASETS = {
    "DS0002": (16101265, "Kraemer2026A-DS0002-Part2-F-D.zip", [5]),
    "DS0004": (16104477, "Kraemer2026A-DS0004-Part2-F-D.zip", [1]),
    "DS0005": (16110531, "Kraemer2026A-DS0005-Part2-F-D.zip", [3]),
    "DS0006": (16139176, "Kraemer2026A-DS0006-Part2-F-D.zip", [2, 36]),
    "DS0007": (16141730, "Kraemer2026A-DS0007-Part2-F-D.zip", [7]),
    "DS0008": (16144559, "Kraemer2026A-DS0008-Part2-F-D.zip", [6]),
    "DS0009": (16147573, "Kraemer2026A-DS0009-Part2-F-D.zip", [16]),
    "DS0010": (16149318, "Kraemer2026A-DS0010-Part2-F-D.zip", [3]),
    "DS0011": (16150544, "Kraemer2026A-DS0011-Part2-F-D.zip", [1]),
    "DS0012": (16151472, "Kraemer2026A-DS0012-Part2-F-D.zip", [2]),
    "DS0013": (16153432, "Kraemer2026A-DS0013-Part2-F-D.zip", [9]),
    "DS0014": (16158358, "Kraemer2026A-DS0014-Part2-F-D.zip", [2]),
    "DS0015": (16159538, "Kraemer2026A-DS0015-Part2-F-D.zip", [9]),
    "DS0016": (16161996, "Kraemer2026A-DS0016-Part2-F-D.zip", [1]),
}
"""Mapping from the name of a dataset to the Zenodo record and file with its raw data and to its labeled time points."""

UNASSIGNED_ID = 65535


def _read_zip_entries(url, cache_path):
    import requests

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

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
        method, crc, csize, usize = fields[4], fields[7], fields[8], fields[9]
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
        entries[name] = {"method": method, "crc": crc, "compressed_size": csize, "header_offset": header_offset}
        pos += 46 + name_len + extra_len + comment_len

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
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
    data = zlib.decompress(response.content, -15) if entry["method"] == 8 else response.content
    if zlib.crc32(data) != entry["crc"]:
        raise RuntimeError("The CRC32 of a downloaded archive member does not match, please try again.")
    return data


def _convert_volume(name, time_point, path, raw_entries, segmentation_entries):
    import h5py
    import tifffile

    out_path = os.path.join(path, "preprocessed", f"{name}_TP{time_point:04d}.h5")
    if os.path.exists(out_path):
        return

    record, raw_file, _ = DATASETS[name]
    stem = f"Kraemer2026A-{name}TP{time_point:04d}DR(D1)CH0001PL"
    raw_member = f"(D1)-ZStacks-Decon-ZS/CH0001/DR(D1)/{stem}(ZS).TIF"
    label_member = f"Kraemer2026A-{name}-Segmentation/{stem}(YD).TIF"
    raw_url = f"{URL_BASE}/{record}/files/{raw_file}/content"
    label_url = f"{URL_BASE}/{SEGMENTATION_RECORD}/files/Kraemer2026A-{name}-Segmentation.zip/content"

    raw = tifffile.imread(io.BytesIO(_read_zip_member(raw_url, raw_entries[raw_member])))
    labels = tifffile.imread(io.BytesIO(_read_zip_member(label_url, segmentation_entries[label_member])))
    labels = np.flip(labels.transpose(1, 0, 2), 0)
    assert raw.shape == labels.shape, f"{name} TP{time_point}: {raw.shape} != {labels.shape}"

    labels = np.where(labels == UNASSIGNED_ID, 0, labels)
    present = np.unique(labels)
    present = present[present > 0]
    lut = np.zeros(UNASSIGNED_ID + 1, dtype="uint16")
    lut[present] = np.arange(1, len(present) + 1)
    labels = lut[labels]

    chunks = (64, 128, 128)
    tmp_path = f"{out_path}.{uuid.uuid4().hex}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw", data=raw, chunks=chunks, compression="gzip")
        f.create_dataset("labels", data=labels, chunks=chunks, compression="gzip")
    os.replace(tmp_path, out_path)


def _validate(datasets):
    datasets = list(DATASETS) if datasets is None else list(datasets)
    invalid = [name for name in datasets if name not in DATASETS]
    if invalid:
        raise ValueError(f"{invalid} are not valid datasets. Choose from {list(DATASETS)}.")
    return datasets


def get_slice2_data(
    path: Union[os.PathLike, str],
    datasets: Optional[Sequence[str]] = None,
    n_workers: int = 2,
    download: bool = False,
) -> str:
    """Download the SLICE-2 dataset and convert the labeled volumes to hdf5 files.

    NOTE: Each volume takes about 0.55 GB (raw) to download. Use `datasets` to only download a subset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        datasets: The names of the datasets to use, see `DATASETS`. By default all 14 are used.
        n_workers: The number of parallel download and conversion workers. A worker needs about 3 GB of memory.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the converted data is stored.
    """
    datasets = _validate(datasets)
    preprocessed_dir = os.path.join(path, "preprocessed")
    pending = [
        (name, time_point) for name in datasets for time_point in DATASETS[name][2]
        if not os.path.exists(os.path.join(preprocessed_dir, f"{name}_TP{time_point:04d}.h5"))
    ]
    if not pending:
        return preprocessed_dir
    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    os.makedirs(preprocessed_dir, exist_ok=True)
    entries_dir = os.path.join(path, "entries")
    raw_entries, segmentation_entries = {}, {}
    for name in sorted({name for name, _ in pending}):
        record, raw_file, _ = DATASETS[name]
        raw_entries[name] = _read_zip_entries(
            f"{URL_BASE}/{record}/files/{raw_file}/content", os.path.join(entries_dir, f"{name}_raw.json")
        )
        segmentation_entries[name] = _read_zip_entries(
            f"{URL_BASE}/{SEGMENTATION_RECORD}/files/Kraemer2026A-{name}-Segmentation.zip/content",
            os.path.join(entries_dir, f"{name}_segmentation.json"),
        )

    with futures.ThreadPoolExecutor(n_workers) as pool:
        tasks = [
            pool.submit(_convert_volume, name, time_point, path, raw_entries[name], segmentation_entries[name])
            for name, time_point in pending
        ]
        for task in tqdm(futures.as_completed(tasks), total=len(tasks), desc="Download SLICE-2 volumes"):
            task.result()

    return preprocessed_dir


def get_slice2_paths(
    path: Union[os.PathLike, str],
    datasets: Optional[Sequence[str]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the SLICE-2 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        datasets: The names of the datasets to use, see `DATASETS`. By default all 14 are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the image data ('raw') and the label data ('labels').
    """
    datasets = _validate(datasets)
    data_dir = get_slice2_data(path, datasets, download=download)
    data_paths = natsorted([
        p for p in glob(os.path.join(data_dir, "*.h5")) if os.path.basename(p).split("_")[0] in datasets
    ])
    assert len(data_paths) > 0
    return data_paths


def get_slice2_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    datasets: Optional[Sequence[str]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SLICE-2 dataset for 3D nucleus segmentation in light-sheet microscopy of Tribolium embryos.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        datasets: The names of the datasets to use, see `DATASETS`. By default all 14 are used.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    data_paths = get_slice2_paths(path, datasets, download)

    kwargs = util.ensure_transforms(ndim=3, **kwargs)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="raw",
        label_paths=data_paths,
        label_key="labels",
        patch_shape=patch_shape,
        ndim=3,
        **kwargs
    )


def get_slice2_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    datasets: Optional[Sequence[str]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SLICE-2 dataloader for 3D nucleus segmentation in light-sheet microscopy of Tribolium embryos.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        datasets: The names of the datasets to use, see `DATASETS`. By default all 14 are used.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_slice2_dataset(
        path, patch_shape, datasets, offsets=offsets, boundaries=boundaries, binary=binary, download=download,
        **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
