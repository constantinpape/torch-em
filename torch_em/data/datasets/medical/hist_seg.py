"""The HIST-Seg dataset contains annotations for orthopedic tissue segmentation in hyperspectral surgical images.

The dataset consists of 220 hyperspectral cubes of ten bovine specimens (22 per specimen), acquired with a HinaLea
camera in a simulated orthopedic surgery setting. Each cube has 299 spectral bands and 608 x 968 pixels and is
paired with a pixel-wise semantic mask with the classes given in `CLASS_NAMES`. NOTE: The label ids follow the
README of the dataset, so bone has the id 0 and there is no unlabeled id, the background of the operative field is
class 6.

`get_hist_seg_data` converts every acquisition to a single hdf5 file. It contains the cube as 'raw' (bands first, shape
(299, 608, 968), float32 reflectance) and the class map as 'labels' (shape (608, 968)). The masks are distributed as
RGB images and are mapped to the class ids with the color table of the README.

The cubes are stored in one ~15.5 GB zip archive per specimen (~155 GB in total). To support downloading only a
subset, single acquisitions are read from the archives with HTTP range requests (each member is verified with its
CRC32) instead of downloading the archives as a whole. Use `subjects` or `n_cases` to select a subset.

NOTE: The dataset page sets a file access request flag, but the files are not restricted and can be downloaded
anonymously.

The data is located at https://doi.org/10.57745/5N62WB, released under a CC-BY-4.0 license.
The code of the authors is available at https://github.com/lsllabisen/HIST-Seg.
Please cite the dataset if you use it for your research.
"""

import os
import json
import uuid
import zlib
import struct
from io import BytesIO
from natsort import natsorted
from concurrent import futures
from typing import Union, Tuple, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


DATAFILE_URL = "https://entrepot.recherche.data.gouv.fr/api/access/datafile/{}"
DATAFILE_IDS = {
    "B0001": 717111, "B0002": 717113, "B0003": 717132, "B0004": 717414, "B0005": 717496,
    "B0006": 717582, "B0007": 717597, "B0008": 717786, "B0009": 717820, "B0010": 717893,
}
SUBJECTS = list(DATAFILE_IDS)

CLASS_NAMES = ["bone", "cartilage", "ligament", "flesh", "fat", "instruments", "background"]
"""The names of the classes. The label id of a class is its index in this list."""

COLORS = [
    (142, 223, 124), (111, 225, 243), (245, 245, 63), (246, 58, 90), (243, 172, 99), (168, 109, 109), (169, 169, 172),
]
BLOCK_SIZE = 64 * 1024 * 1024


def _resolve(datafile_id):
    import requests

    headers = {"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"}
    response = requests.get(DATAFILE_URL.format(datafile_id), headers=headers, allow_redirects=True)
    response.raise_for_status()
    return response.url, int(response.headers["Content-Range"].split("/")[1])


def _get_range(url, start, end):
    import requests

    response = requests.get(url, headers={"Range": f"bytes={start}-{end}"})
    response.raise_for_status()
    return response.content


def _read_zip_entries(url, size):
    tail = _get_range(url, size - 65536, size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    n_entries, cd_size, cd_offset = struct.unpack("<HII", tail[eocd + 10:eocd + 20])
    if n_entries == 0xFFFF or cd_offset == 0xFFFFFFFF:
        locator = tail.rfind(b"PK\x06\x07")
        zip64_offset = struct.unpack("<Q", tail[locator + 8:locator + 16])[0]
        zip64 = _get_range(url, zip64_offset, zip64_offset + 55)
        n_entries, cd_size, cd_offset = struct.unpack("<QQQ", zip64[32:56])
    directory = _get_range(url, cd_offset, cd_offset + cd_size - 1)

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

    return entries


def _get_entries(path, subject, download):
    cache_path = os.path.join(path, f"entries_{subject}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    url, size = _resolve(DATAFILE_IDS[subject])
    entries = _read_zip_entries(url, size)
    tmp_path = f"{cache_path}.{uuid.uuid4().hex}.incomplete"
    with open(tmp_path, "w") as f:
        json.dump(entries, f)
    os.replace(tmp_path, cache_path)
    return entries


def _read_zip_member(url, entry):
    offset = entry["header_offset"]
    header = _get_range(url, offset, offset + 29)
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    start = offset + 30 + name_len + extra_len

    chunks = []
    for block_start in range(0, entry["compressed_size"], BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, entry["compressed_size"]) - 1
        chunks.append(_get_range(url, start + block_start, start + block_end))
    data = b"".join(chunks)

    if entry["method"] == 8:
        data = zlib.decompress(data, -15)
    if zlib.crc32(data) != entry["crc"]:
        raise RuntimeError("The checksum of a downloaded archive member does not match.")
    return data


def _parse_header(header):
    fields = {}
    for line in header.decode(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    assert fields["interleave"] == "bsq" and fields["data type"] == "4" and fields["byte order"] == "0", fields
    return int(fields["bands"]), int(fields["lines"]), int(fields["samples"])


def _mask_to_labels(mask):
    encoded = (mask[..., 0].astype("uint32") << 16) | (mask[..., 1].astype("uint32") << 8) | mask[..., 2]
    labels = np.full(encoded.shape, 255, dtype="uint8")
    for class_id, (r, g, b) in enumerate(COLORS):
        labels[encoded == ((r << 16) | (g << 8) | b)] = class_id
    assert (labels != 255).all(), "The mask contains a color that does not belong to a class."
    return labels


def _convert_acquisition(subject, acquisition, path, entries):
    import h5py
    from PIL import Image

    out_path = os.path.join(path, "preprocessed", f"{acquisition}.h5")
    if os.path.exists(out_path):
        return

    url, _ = _resolve(DATAFILE_IDS[subject])
    prefix = f"{subject}/{acquisition}"
    bands, lines, samples = _parse_header(_read_zip_member(url, entries[f"{prefix}/{acquisition}.hdr"]))
    cube = np.frombuffer(_read_zip_member(url, entries[f"{prefix}/{acquisition}.dat"]), dtype="<f4")
    cube = cube.reshape(bands, lines, samples)

    mask = np.array(Image.open(BytesIO(_read_zip_member(url, entries[f"{prefix}/Annotations/{acquisition}.png"]))))
    labels = _mask_to_labels(mask[..., :3])
    assert labels.shape == cube.shape[1:], f"{acquisition}: {labels.shape} != {cube.shape[1:]}"

    tmp_path = f"{out_path}.{uuid.uuid4().hex}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw", data=cube, chunks=(bands, 64, 64), compression="gzip")
        f.create_dataset("labels", data=labels, chunks=(64, 64), compression="gzip")
    os.replace(tmp_path, out_path)


def _list_acquisitions(path, subjects, n_cases, download):
    acquisitions = []
    for subject in subjects:
        entries = _get_entries(path, subject, download)
        names = natsorted({name.split("/")[1] for name in entries if name.endswith(".dat")})
        acquisitions.extend(
            (subject, name) for name in names if f"{subject}/{name}/Annotations/{name}.png" in entries
        )
        if n_cases is not None and len(acquisitions) >= n_cases:
            break
    return acquisitions[:n_cases]


def get_hist_seg_data(
    path: Union[os.PathLike, str],
    subjects: Optional[Sequence[str]] = None,
    n_cases: Optional[int] = None,
    n_workers: int = 2,
    download: bool = False,
) -> str:
    """Download the HIST-Seg dataset and convert the acquisitions to hdf5 files.

    NOTE: The full collection is about 155 GB. Use `subjects` or `n_cases` to only download a subset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        subjects: The specimens to use, a subset of `SUBJECTS`. By default all ten specimens are used.
        n_cases: The number of acquisitions to use, sorted by specimen and acquisition id. By default all are used.
        n_workers: The number of parallel download and conversion workers. Each worker holds a cube of ~0.7 GB.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the converted hdf5 files.
    """
    subjects = SUBJECTS if subjects is None else list(subjects)
    invalid = [s for s in subjects if s not in DATAFILE_IDS]
    if invalid:
        raise ValueError(f"{invalid} are not valid subjects. Choose from {SUBJECTS}.")

    os.makedirs(path, exist_ok=True)
    preprocessed_dir = os.path.join(path, "preprocessed")
    acquisitions = _list_acquisitions(path, subjects, n_cases, download)

    missing = [(s, a) for s, a in acquisitions if not os.path.exists(os.path.join(preprocessed_dir, f"{a}.h5"))]
    if not missing:
        return preprocessed_dir
    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    os.makedirs(preprocessed_dir, exist_ok=True)
    entries = {subject: _get_entries(path, subject, download) for subject in {s for s, _ in missing}}
    with futures.ThreadPoolExecutor(n_workers) as pool:
        tasks = [pool.submit(_convert_acquisition, s, a, path, entries[s]) for s, a in missing]
        for task in tqdm(futures.as_completed(tasks), total=len(tasks), desc="Download and convert HIST-Seg"):
            task.result()

    return preprocessed_dir


def get_hist_seg_paths(
    path: Union[os.PathLike, str],
    subjects: Optional[Sequence[str]] = None,
    n_cases: Optional[int] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the HIST-Seg data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        subjects: The specimens to use, a subset of `SUBJECTS`. By default all ten specimens are used.
        n_cases: The number of acquisitions to use, sorted by specimen and acquisition id. By default all are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the hdf5 files, which contain the cubes ('raw') and the class maps ('labels').
    """
    subjects = SUBJECTS if subjects is None else list(subjects)
    preprocessed_dir = get_hist_seg_data(path, subjects, n_cases, download=download)
    acquisitions = _list_acquisitions(path, subjects, n_cases, download)
    return [os.path.join(preprocessed_dir, f"{a}.h5") for _, a in acquisitions]


def get_hist_seg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    subjects: Optional[Sequence[str]] = None,
    n_cases: Optional[int] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HIST-Seg dataset for tissue segmentation in hyperspectral surgical images.

    The raw data has 299 channels (the spectral bands), the labels are the class ids of `CLASS_NAMES`.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        subjects: The specimens to use, a subset of `SUBJECTS`. By default all ten specimens are used.
        n_cases: The number of acquisitions to use, sorted by specimen and acquisition id. By default all are used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_hist_seg_paths(path, subjects, n_cases, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_hist_seg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    subjects: Optional[Sequence[str]] = None,
    n_cases: Optional[int] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the HIST-Seg dataloader for tissue segmentation in hyperspectral surgical images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        subjects: The specimens to use, a subset of `SUBJECTS`. By default all ten specimens are used.
        n_cases: The number of acquisitions to use, sorted by specimen and acquisition id. By default all are used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hist_seg_dataset(path, patch_shape, subjects, n_cases, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
