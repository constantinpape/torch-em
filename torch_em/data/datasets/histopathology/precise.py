"""The PRECISE dataset contains annotations for the semantic segmentation of prostate tissue and lesions
in paired H&E and immunohistochemistry (IHC) whole-slide images of prostate core needle biopsies.

PRECISE (PRostate Expert-annotated Contiguous IHC-H&E Serial sEctions) consists of 54 slide pairs from 25 patients
(sub-01 has three sessions, all other patients one). Each pair has a H&E and a HMWCK-AMACR (CKAPM + racemase) IHC
whole-slide image, scanned at 0.243 micrometer per pixel with a 3DHISTECH Pannoramic scanner, and a pixel-level
mask for each stain, drawn by expert uropathologists (24,387 annotations in total).

The label ids of the masks are:
- 0: background
- 1: tumor
- 2: benign gland
- 3: artifact
- 4: high-grade prostatic intraepithelial neoplasia (HGPIN)
- 5: intraductal carcinoma
- 6: atypical intraductal proliferation
- 7: stroma
Not every class occurs in every slide.

The data is located at https://doi.org/10.5281/zenodo.20721779 as a single 55.6 GB zip archive with pyramidal
OME-TIFF images and masks, released under a CC BY 4.0 license. To avoid downloading the whole archive, this module reads
its directory from the server and fetches only the requested slides and masks. Each slide is converted once into a
chunked HDF5 file at a chosen pyramid level (pixel size 0.243 * 2 ** level micrometer, the default level 1 is
0.486 micrometer), the downloaded OME-TIFFs are removed afterwards. Use `n_cases` to only prepare a subset.

The publication for this dataset was not available at the time of implementation, so please cite the Zenodo record
if you use it in your research.
"""

import os
import re
import json
import uuid
import zlib
import struct
from typing import List, Literal, Optional, Tuple, Union

from tqdm import tqdm

import torch

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


RECORD_URL = "https://zenodo.org/api/records/20721779/files"
ZIP_URL = f"{RECORD_URL}/data.zip/content"

SMALL_FILES = {
    "label_descriptions.json": "31f9687f2611fce5a56e5f983c9a97b6ae0f5d1b6171be65e9bbc961ad2ec11c",
    "participants.csv": "8b5f18120b83e84eb235eb746eb005f8dbbd0b9d46c30d248c90c0a7b31cdcaf",
}

STAINS = {"he": "h-e", "ihc": "hmwck-amacr"}

CLASS_NAMES = {
    0: "background",
    1: "tumor",
    2: "benign gland",
    3: "artifact",
    4: "high-grade prostatic intraepithelial neoplasia",
    5: "intraductal carcinoma",
    6: "atypical intraductal proliferation",
    7: "stroma",
}

N_LEVELS = 6


def _range_get(url, start, end, **kwargs):
    import requests

    response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, **kwargs)
    response.raise_for_status()
    return response


def _read_zip_entries(path):
    cache_path = os.path.join(path, "zip_entries.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    import requests

    size = int(requests.head(ZIP_URL, allow_redirects=True).headers["Content-Length"])
    tail = _range_get(ZIP_URL, size - 65557, size - 1).content
    eocd = tail[tail.rfind(b"PK\x05\x06"):]
    n_entries, cd_size, cd_offset = struct.unpack("<HII", eocd[10:20])
    if cd_offset == 0xFFFFFFFF or n_entries == 0xFFFF:
        locator = tail[tail.rfind(b"PK\x06\x07"):][:20]
        zip64_offset = struct.unpack("<Q", locator[8:16])[0]
        record = _range_get(ZIP_URL, zip64_offset, zip64_offset + 55).content
        n_entries, cd_size, cd_offset = struct.unpack("<QQQ", record[32:56])

    directory = _range_get(ZIP_URL, cd_offset, cd_offset + cd_size - 1).content
    entries, pos = {}, 0
    while pos < len(directory):
        (signature, _, _, _, _, _, _, crc, comp_size, size, name_len, extra_len, comment_len, _, _, _,
         header_offset) = struct.unpack("<IHHHHHHIIIHHHHHII", directory[pos:pos + 46])
        assert signature == 0x02014B50, "The zip directory could not be parsed."
        name = directory[pos + 46:pos + 46 + name_len].decode()
        extra = directory[pos + 46 + name_len:pos + 46 + name_len + extra_len]
        extra_pos = 0
        while extra_pos < len(extra):
            field_id, field_size = struct.unpack("<HH", extra[extra_pos:extra_pos + 4])
            if field_id == 1:
                field, k = extra[extra_pos + 4:extra_pos + 4 + field_size], 0
                if size == 0xFFFFFFFF:
                    size, k = struct.unpack("<Q", field[k:k + 8])[0], k + 8
                if comp_size == 0xFFFFFFFF:
                    comp_size, k = struct.unpack("<Q", field[k:k + 8])[0], k + 8
                if header_offset == 0xFFFFFFFF:
                    header_offset = struct.unpack("<Q", field[k:k + 8])[0]
            extra_pos += 4 + field_size
        if size > 0:
            entries[name] = {"crc": crc, "comp_size": comp_size, "size": size, "offset": header_offset}
        pos += 46 + name_len + extra_len + comment_len

    tmp_path = f"{cache_path}.{uuid.uuid4().hex}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(entries, f)
    os.replace(tmp_path, cache_path)
    return entries


def _fetch_member(entry, out_path):
    header = _range_get(ZIP_URL, entry["offset"], entry["offset"] + 29).content
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    start = entry["offset"] + 30 + name_len + extra_len

    decompressor, crc, n_bytes = zlib.decompressobj(-15), 0, 0
    tmp_path = f"{out_path}.{uuid.uuid4().hex}.tmp"
    with _range_get(ZIP_URL, start, start + entry["comp_size"] - 1, stream=True) as response, \
            open(tmp_path, "wb") as f:
        for chunk in response.iter_content(1 << 20):
            data = decompressor.decompress(chunk)
            crc, n_bytes = zlib.crc32(data, crc), n_bytes + len(data)
            f.write(data)
        data = decompressor.flush()
        crc, n_bytes = zlib.crc32(data, crc), n_bytes + len(data)
        f.write(data)

    if crc != entry["crc"] or n_bytes != entry["size"]:
        os.remove(tmp_path)
        raise RuntimeError(f"The download of {os.path.basename(out_path)} is corrupted, please try again.")
    os.replace(tmp_path, out_path)


def _get_cases(entries, stain):
    stain_dir = STAINS[stain]
    stain = re.escape(stain_dir)
    pattern = re.compile(rf"^data/sub-\d+/ses-\d+/wsi_{stain}/(sub-\d+_ses-\d+)_{stain}\.ome\.tif$")
    cases = {}
    for name in entries:
        match = pattern.match(name)
        if match:
            case = match.group(1)
            mask_name = name.replace(".ome.tif", "_mask.ome.tif")
            assert mask_name in entries, f"Cannot find the mask for {name}."
            cases[case] = (name, mask_name)
    return dict(sorted(cases.items()))


def _open_level(series, level_index):
    import zarr

    # The pyramidal TIFFs are tiled, so the zarr view reads only the requested tiles.
    array = zarr.open(series.aszarr(), mode="r")
    return array if hasattr(array, "shape") else array[str(level_index)]


def _convert_case(image_path, mask_path, out_path, level, tile=4096):
    import h5py
    import tifffile
    import numpy as np

    image_series = tifffile.TiffFile(image_path).series[0]
    mask_series = tifffile.TiffFile(mask_path).series[0]
    image, mask = _open_level(image_series, level), _open_level(mask_series, level)

    # The pyramid levels of the image and mask are rounded differently, so the mask level can be a few pixels off.
    # It is mapped to the image grid by nearest neighbor sampling, unless the shapes really differ.
    height, width = image.shape[:2]
    if any(abs(image.shape[i] - mask.shape[i]) > 1e-3 * image.shape[i] + 1 for i in range(2)):
        raise RuntimeError(f"The image {image.shape} and mask {mask.shape} shapes of {image_path} do not match.")
    rows = np.minimum(((np.arange(height) + 0.5) * mask.shape[0] / height).astype(int), mask.shape[0] - 1)
    cols = np.minimum(((np.arange(width) + 0.5) * mask.shape[1] / width).astype(int), mask.shape[1] - 1)

    tmp_path = f"{out_path}.{uuid.uuid4().hex}.tmp"
    with h5py.File(tmp_path, "w") as f:
        raw = f.create_dataset(
            "raw", shape=(3, height, width), dtype="uint8", compression="gzip", chunks=(3, 512, 512)
        )
        labels = f.create_dataset(
            "labels", shape=(height, width), dtype="uint8", compression="gzip", chunks=(512, 512)
        )
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                bb = (slice(y, min(y + tile, height)), slice(x, min(x + tile, width)))
                raw[(slice(None),) + bb] = image[bb].transpose(2, 0, 1)
                r, c = rows[bb[0]], cols[bb[1]]
                mask_block = mask[r[0]:r[-1] + 1, c[0]:c[-1] + 1]
                labels[bb] = mask_block[np.ix_(r - r[0], c - c[0])]
    os.replace(tmp_path, out_path)


def get_precise_data(
    path: Union[os.PathLike, str],
    stain: Literal["he", "ihc"] = "he",
    n_cases: Optional[int] = None,
    level: int = 1,
    download: bool = False,
) -> str:
    """Download and preprocess the PRECISE dataset.

    NOTE: The full archive is 55.6 GB and is never downloaded as a whole, only the requested slides are fetched.
    Use `n_cases` to only prepare a subset, e.g. a slide takes about 0.5 to 1.4 GB to download.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        stain: The choice of stain. Either 'he' (H&E) or 'ihc' (HMWCK-AMACR IHC).
        n_cases: The number of slides to use, sorted by slide id ('sub-<patient>_ses-<session>').
            By default all 54 slides of the stain are used.
        level: The pyramid level of the preprocessed data (0 to 5), with a pixel size of 0.243 * 2 ** level
            micrometer. The preprocessed data is stored separately for each level.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the preprocessed data is stored.
    """
    if stain not in STAINS:
        raise ValueError(f"'{stain}' is not a valid stain. Choose one of {list(STAINS)}.")
    if level not in range(N_LEVELS):
        raise ValueError(f"'{level}' is not a valid pyramid level. Choose a level from 0 to {N_LEVELS - 1}.")

    os.makedirs(path, exist_ok=True)
    for filename, checksum in SMALL_FILES.items():
        util.download_source(
            path=os.path.join(path, filename), url=f"{RECORD_URL}/{filename}/content", download=download,
            checksum=checksum,
        )

    if not download and not os.path.exists(os.path.join(path, "zip_entries.json")):
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")
    entries = _read_zip_entries(path)
    cases = _get_cases(entries, stain)
    case_ids = list(cases)[:n_cases]

    preprocessed_dir = os.path.join(path, "preprocessed", f"level{level}", stain)
    raw_dir = os.path.join(path, "raw", stain)
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    missing = [case for case in case_ids if not os.path.exists(os.path.join(preprocessed_dir, f"{case}.h5"))]
    if missing and not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    for case in tqdm(missing, desc="Prepare PRECISE slides"):
        image_name, mask_name = cases[case]
        image_path, mask_path = (os.path.join(raw_dir, os.path.basename(name)) for name in (image_name, mask_name))
        for name, out_path in ((image_name, image_path), (mask_name, mask_path)):
            if not os.path.exists(out_path):
                _fetch_member(entries[name], out_path)

        _convert_case(image_path, mask_path, os.path.join(preprocessed_dir, f"{case}.h5"), level)
        os.remove(image_path)
        os.remove(mask_path)

    return preprocessed_dir


def get_precise_paths(
    path: Union[os.PathLike, str],
    stain: Literal["he", "ihc"] = "he",
    n_cases: Optional[int] = None,
    level: int = 1,
    download: bool = False,
) -> List[str]:
    """Get paths to the PRECISE data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        stain: The choice of stain. Either 'he' (H&E) or 'ihc' (HMWCK-AMACR IHC).
        n_cases: The number of slides to use, sorted by slide id ('sub-<patient>_ses-<session>').
            By default all 54 slides of the stain are used.
        level: The pyramid level of the preprocessed data (0 to 5), with a pixel size of 0.243 * 2 ** level
            micrometer.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the preprocessed HDF5 files, which contain the image data ('raw') and the
        label data ('labels').
    """
    preprocessed_dir = get_precise_data(path, stain, n_cases, level, download)
    case_ids = list(_get_cases(_read_zip_entries(path), stain))[:n_cases]
    return [os.path.join(preprocessed_dir, f"{case}.h5") for case in case_ids]


def get_precise_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    stain: Literal["he", "ihc"] = "he",
    n_cases: Optional[int] = None,
    level: int = 1,
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> Dataset:
    """Get the PRECISE dataset for semantic segmentation of prostate tissue and lesions in whole-slide images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        stain: The choice of stain. Either 'he' (H&E) or 'ihc' (HMWCK-AMACR IHC).
        n_cases: The number of slides to use, sorted by slide id ('sub-<patient>_ses-<session>').
            By default all 54 slides of the stain are used.
        level: The pyramid level of the preprocessed data (0 to 5), with a pixel size of 0.243 * 2 ** level
            micrometer.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of the labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_precise_paths(path, stain, n_cases, level, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key="labels",
        patch_shape=patch_shape,
        label_dtype=label_dtype,
        is_seg_dataset=True,
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_precise_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    stain: Literal["he", "ihc"] = "he",
    n_cases: Optional[int] = None,
    level: int = 1,
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PRECISE dataloader for semantic segmentation of prostate tissue and lesions in whole-slide images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        stain: The choice of stain. Either 'he' (H&E) or 'ihc' (HMWCK-AMACR IHC).
        n_cases: The number of slides to use, sorted by slide id ('sub-<patient>_ses-<session>').
            By default all 54 slides of the stain are used.
        level: The pyramid level of the preprocessed data (0 to 5), with a pixel size of 0.243 * 2 ** level
            micrometer.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of the labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_precise_dataset(
        path=path, patch_shape=patch_shape, stain=stain, n_cases=n_cases, level=level, download=download,
        label_dtype=label_dtype, resize_inputs=resize_inputs, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
