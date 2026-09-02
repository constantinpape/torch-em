"""This dataset contains annotations for prostate epithelium segmentation in
H&E-stained whole-slide histopathology images of prostatectomy specimens.

The data is from the publication https://doi.org/10.1038/s41598-018-37257-4
("Epithelium segmentation using deep learning in H&E-stained prostate specimens
with immunohistochemistry as reference standard"). It is hosted on Zenodo at
https://zenodo.org/records/1485967 under a CC BY-NC-SA 4.0 license.
Please cite the publication if you use this dataset.

This loader exposes the 25 training whole-slide images whose epithelium mask
received manual pathologist correction. The remaining 37 training masks are
uncorrected predictions of an IHC-guided U-Net, and the 40 test slides ship
only region outlines and a benign/cancer label rather than a raster mask, so
neither is included here.

Mask label values: 0 (unannotated), 1 (non-epithelium tissue within an
annotated region), and 2 (epithelium). Only small regions of interest within
each slide are annotated (about 7% of a slide's area); the rest of a slide's
mask is unannotated, so `get_peso_dataset` / `get_peso_loader` default to a
`MinForegroundSampler` that rejects mostly-unannotated patches.

NOTE: The whole-slide images and masks are multi-resolution pyramidal TIFFs of
several gigabytes each, bundled inside a few large multi-slide zip archives on
Zenodo. To avoid downloading unrelated slides, each requested slide is fetched
as a single zip member via an HTTP range request rather than the whole archive.
On first use each requested slide is also converted into a chunked HDF5 file at
the requested pyramid level, which requires some time and disk space.
"""

import os
import struct
import zipfile
import zlib
from pathlib import Path
from typing import List, Optional, Tuple, Union

from tqdm import tqdm

import requests

import torch
from torch.utils.data import Dataset, DataLoader

import torch_em
from torch_em.data.sampler import MinForegroundSampler

from .. import util


RECORD_URL = "https://zenodo.org/api/records/1485967/files"

CHECKSUMS = {  # md5 of the full Zenodo archives, kept for provenance
    "peso_training_masks_corrected.zip": "8e2c86fcecfafe09c9d48a60b42441b5",
    "peso_training_wsi_1.zip": "8e4e53d7ba855fc2f318dce94b05fe31",
    "peso_training_wsi_2.zip": "bfcae8b444c12c0ecbb717dc37334020",
    "peso_training_wsi_3.zip": "f7d484acec429c3a9d9e685969edd82b",
    "peso_training_wsi_4.zip": "75a350b450193b48e9bed3a3484f639d",
    "peso_training_wsi_5.zip": "bc94373db95c5e2ceefe08c45a8e54db",
    "peso_training_wsi_6.zip": "4fa8cdd4b748d67b6c39a982be7b627a",
}

# The 25 corrected slides, mapped to the wsi archive that holds their raw image.
STEM_TO_WSI_ZIP = {
    "pds_6": 1, "pds_8": 1,
    "pds_34": 2,
    "pds_35": 3, "pds_38": 3, "pds_39": 3, "pds_40": 3, "pds_43": 3,
    "pds_46": 4, "pds_56": 4, "pds_60": 4, "pds_64": 4,
    "pds_69": 5, "pds_70": 5, "pds_71": 5, "pds_72": 5, "pds_73": 5, "pds_79": 5,
    "pds_91": 6, "pds_93": 6, "pds_96": 6, "pds_99": 6, "pds_100": 6, "pds_101": 6, "pds_102": 6,
}


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


def _download_zip_member(archive_url, member, dst_path, chunk_size=1024 * 1024 * 8):
    """Fetch one member of a large remote zip via a single HTTP range request for its
    compressed bytes, instead of downloading the whole multi-slide archive.
    """
    if os.path.exists(dst_path):
        return

    size = int(requests.head(archive_url, allow_redirects=True).headers["Content-Length"])
    zf = zipfile.ZipFile(_RemoteZipReader(archive_url, size))
    info = zf.getinfo(member)

    header_reader = _RemoteZipReader(archive_url, size)
    header_reader.seek(info.header_offset)
    local_header = header_reader.read(30)
    fn_len, extra_len = struct.unpack("<HH", local_header[26:30])
    data_start = info.header_offset + 30 + fn_len + extra_len
    data_end = data_start + info.compress_size - 1

    part_path = dst_path + ".part"
    with requests.get(archive_url, headers={"Range": f"bytes={data_start}-{data_end}"}, stream=True) as r:
        r.raise_for_status()
        with open(part_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)

    if info.compress_type == zipfile.ZIP_STORED:
        os.replace(part_path, dst_path)
        return

    decompressor = zlib.decompressobj(-15)
    with open(part_path, "rb") as src, open(dst_path, "wb") as dst:
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            dst.write(decompressor.decompress(chunk))
        dst.write(decompressor.flush())
    os.remove(part_path)


def _resolve_sample_ids(sample_ids):
    stems = sorted(STEM_TO_WSI_ZIP, key=lambda s: int(s.split("_")[1]))
    if sample_ids is None:
        return stems
    missing = sorted(set(sample_ids) - set(stems))
    if missing:
        raise ValueError(f"The following sample ids are not part of this dataset: {missing}")
    return [stem for stem in stems if stem in sample_ids]


def _open_level(series, level_index):
    import zarr

    # The pyramidal TIFFs are natively tiled, so a zarr view reads only the requested tiles lazily.
    array = zarr.open(series.aszarr(), mode="r")
    return array if hasattr(array, "shape") else array[str(level_index)]


def _convert_slide(image_path, mask_path, output_path, resolution_level, tile=4096):
    import h5py
    import tifffile

    image_series = tifffile.TiffFile(image_path).series[0]
    mask_series = tifffile.TiffFile(mask_path).series[0]

    # The released masks were rasterized one pyramid level finer than the raw slide.
    mask_level = resolution_level + 1
    height, width = image_series.levels[resolution_level].shape[:2]
    mask_height, mask_width = mask_series.levels[mask_level].shape[:2]
    if abs(height - mask_height) > 1 or abs(width - mask_width) > 1:
        raise RuntimeError(
            f"The mask '{mask_path}' does not match the raw shape ({height}, {width}) "
            f"at level {resolution_level}: got ({mask_height}, {mask_width}) at mask level {mask_level}."
        )
    height, width = min(height, mask_height), min(width, mask_width)

    image = _open_level(image_series, resolution_level)
    mask = _open_level(mask_series, mask_level)

    tmp_path = output_path + ".tmp"
    with h5py.File(tmp_path, "w") as f:
        raw = f.create_dataset(
            "images/raw", shape=(3, height, width), dtype="uint8", compression="gzip", chunks=(1, 512, 512)
        )
        labels = f.create_dataset(
            "labels/mask", shape=(height, width), dtype="uint8", compression="gzip", chunks=(512, 512)
        )
        for y in tqdm(range(0, height, tile), desc=f"Converting {Path(image_path).stem}"):
            for x in range(0, width, tile):
                th, tw = min(tile, height - y), min(tile, width - x)
                raw[:, y:y + th, x:x + tw] = image[y:y + th, x:x + tw].transpose(2, 0, 1)
                labels[y:y + th, x:x + tw] = mask[y:y + th, x:x + tw]

    os.replace(tmp_path, output_path)


def get_peso_data(
    path: Union[os.PathLike, str],
    sample_ids: Optional[List[str]] = None,
    resolution_level: int = 0,
    download: bool = False,
) -> str:
    """Download and preprocess the PESO prostate epithelium segmentation data.

    Args:
        path: Filepath to a folder where the data will be saved.
        sample_ids: The slide stems to restrict the data to, e.g. ['pds_8', 'pds_34'].
            By default all 25 corrected slides are used.
        resolution_level: The pyramid level to convert. 0 is the native raw resolution;
            use a higher level to reduce the size of the preprocessed data.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the preprocessed data is stored.
    """
    stems = _resolve_sample_ids(sample_ids)

    raw_dir = os.path.join(path, "raw")
    mask_dir = os.path.join(path, "masks")
    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    for stem in stems:
        output_path = os.path.join(preprocessed_dir, f"{stem}_level{resolution_level}.h5")
        if os.path.exists(output_path):
            continue

        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        image_path = os.path.join(raw_dir, f"{stem}_HE.tif")
        mask_path = os.path.join(mask_dir, f"{stem}_HE_training_mask_corrected.tif")

        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            if not download:
                raise RuntimeError(f"Data for '{stem}' is not found and download is set to False.")

            wsi_url = f"{RECORD_URL}/peso_training_wsi_{STEM_TO_WSI_ZIP[stem]}.zip/content"
            masks_url = f"{RECORD_URL}/peso_training_masks_corrected.zip/content"
            _download_zip_member(wsi_url, f"{stem}_HE.tif", image_path)
            _download_zip_member(masks_url, f"{stem}_HE_training_mask_corrected.tif", mask_path)

        _convert_slide(image_path, mask_path, output_path, resolution_level)

    return preprocessed_dir


def get_peso_paths(
    path: Union[os.PathLike, str],
    sample_ids: Optional[List[str]] = None,
    resolution_level: int = 0,
    download: bool = False,
) -> List[str]:
    """Get paths to the PESO prostate epithelium segmentation data.

    Args:
        path: Filepath to a folder where the data will be saved.
        sample_ids: The slide stems to restrict the data to, e.g. ['pds_8', 'pds_34'].
            By default all 25 corrected slides are used.
        resolution_level: The pyramid level to convert. 0 is the native raw resolution;
            use a higher level to reduce the size of the preprocessed data.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the preprocessed HDF5 files.
    """
    preprocessed_dir = get_peso_data(path, sample_ids, resolution_level, download)
    stems = _resolve_sample_ids(sample_ids)
    return [os.path.join(preprocessed_dir, f"{stem}_level{resolution_level}.h5") for stem in stems]


def get_peso_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    sample_ids: Optional[List[str]] = None,
    resolution_level: int = 0,
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> Dataset:
    """Get the PESO dataset for prostate epithelium segmentation in whole-slide histopathology images.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape to use for training.
        sample_ids: The slide stems to restrict the data to, e.g. ['pds_8', 'pds_34'].
            By default all 25 corrected slides are used.
        resolution_level: The pyramid level to convert. 0 is the native raw resolution;
            use a higher level to reduce the size of the preprocessed data.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of the labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_peso_paths(path, sample_ids, resolution_level, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    # Only small regions of interest are annotated per slide, so most patches would otherwise be unannotated.
    kwargs.setdefault("sampler", MinForegroundSampler(min_fraction=0.05, background_id=0, p_reject=0.9))

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="images/raw",
        label_paths=volume_paths,
        label_key="labels/mask",
        patch_shape=patch_shape,
        label_dtype=label_dtype,
        is_seg_dataset=True,
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_peso_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    sample_ids: Optional[List[str]] = None,
    resolution_level: int = 0,
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PESO dataloader for prostate epithelium segmentation in whole-slide histopathology images.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        sample_ids: The slide stems to restrict the data to, e.g. ['pds_8', 'pds_34'].
            By default all 25 corrected slides are used.
        resolution_level: The pyramid level to convert. 0 is the native raw resolution;
            use a higher level to reduce the size of the preprocessed data.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of the labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_peso_dataset(
        path=path, patch_shape=patch_shape, sample_ids=sample_ids, resolution_level=resolution_level,
        download=download, label_dtype=label_dtype, resize_inputs=resize_inputs, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
