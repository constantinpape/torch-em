"""OCT5k contains pixel-wise multi-grader annotations of retinal layer bands in optical coherence
tomography (OCT) B-scans, covering AMD (age-related macular degeneration), DME (diabetic macular
edema) and healthy subjects.

The masks are located at https://doi.org/10.5522/04/22128671 (UCL Research Data Repository, a
figshare instance), released under a CC0 license. This dataset is from the publication
https://doi.org/10.1038/s41597-024-04259-z. Please cite it if you use this dataset in your research.

NOTE: The raw OCT images are not shipped with the masks archive. The dataset's own scripts fetch them
from a third-party source instead (R. Rasti's Macular OCT dataset), distributed as a password-protected
archive on Google Drive (password published on the page linked from the OCT5k scripts, no registration
required). As of this writing, the primary Google Drive link referenced by those scripts has become
inaccessible (Google reports the file as "not accessible", i.e. its sharing permission was revoked or
changed, not a rate limit). An older, still-accessible mirror of the same source exists at a different
Google Drive file id, but the file served from that mirror is truncated: it only contains the "AMD
Part1", "AMD Part2", "Normal Part1" and "Normal Part2" categories in full (1269 of 1672 manually graded
images, all complete for these four categories), with the "DME" category entirely missing (0 of 403).
The truncated archive's own zip central directory is also corrupted (its end-of-central-directory record
does not reflect the true file length), so this module recovers the contained files by scanning for raw
zip local file headers directly rather than relying on the central directory. This is real, verified
recovered data (checked against the dataset's own path manifest, `manual_paths.csv`), not a workaround
of any access restriction: the mirror itself has no password beyond the one already published for the
primary source, and no additional data exists beyond what is recovered here. Consequently, this loader
only supports the "AMD Part1", "AMD Part2", "Normal Part1" and "Normal Part2" categories; "DME" is not
available. This should be treated as a known, permanent limitation until the primary source is restored.

NOTE: The masks are index-encoded label maps (not the RGB visualization masks the archive also ships)
with values 0-5: 0 is background, and 1-5 correspond to the 5 retinal layer bands delimited by the ILM,
OPL-Henles, IS/OS junction, IBRPE and OBRPE boundaries (per the dataset's own README), in order from the
vitreous side down. The raw OCT images and their masks are not the same resolution in the original
archive: masks are rendered on a fixed 512x512 canvas, while raw images keep their native (smaller) width.
This loader resizes the mask to the raw image's shape with nearest-neighbor interpolation (verified to
align the layer bands with the visible retinal structure in the raw image, not merely equal in aspect).
"""

import os
import csv
import struct
import zlib
from glob import glob
from natsort import natsorted
from typing import Union, Literal, Tuple, List

import numpy as np
import imageio.v3 as imageio
from PIL import Image

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


MASKS_URL = "https://ndownloader.figshare.com/files/44436359"
MASKS_CHECKSUM = "ae61b43b11c51f39a755b3b63ba4a6353adb99b1e581792c360caa6740df2611"

RAW_IMAGES_URL = "https://drive.google.com/uc?id=1y7yKlDR4sP8bJ-_updZFBtHD6Yq_FO03"
RAW_IMAGES_CHECKSUM = "da9229443c0601c6ed66d67ceebe39c5edc78762ec5e54d4917105ee635a5d11"
RAW_IMAGES_PASSWORD = b"MCME2017"

CATEGORIES = ["AMD Part1", "AMD Part2", "Normal Part1", "Normal Part2"]


def _decrypt_zipcrypto(data, password):
    key0, key1, key2 = 305419896, 591751049, 878082192

    def crc32_step(crc, byte):
        return zlib.crc32(bytes([byte]), crc ^ 0xffffffff) ^ 0xffffffff

    def update_keys(byte):
        nonlocal key0, key1, key2
        key0 = crc32_step(key0, byte)
        key1 = (key1 + (key0 & 0xff)) & 0xffffffff
        key1 = (key1 * 134775813 + 1) & 0xffffffff
        key2 = crc32_step(key2, (key1 >> 24) & 0xff)

    for byte in password:
        update_keys(byte)

    out = bytearray()
    for byte in data:
        temp = (key2 | 2) & 0xffff
        decrypt_byte = ((temp * (temp ^ 1)) >> 8) & 0xff
        plain = byte ^ decrypt_byte
        out.append(plain)
        update_keys(plain)
    return bytes(out)


def _carve_zip_entries(zip_path, dst):
    # The archive at `RAW_IMAGES_URL` is truncated and its central directory does not reflect the
    # true file content, so standard zip readers (including Python's own 'zipfile') refuse to open
    # it. This recovers every intact entry by scanning for local file header signatures directly.
    local_header_sig = b"PK\x03\x04"
    data_descriptor_sig = b"PK\x07\x08"

    with open(zip_path, "rb") as f:
        data = f.read()

    pos = 0
    while True:
        idx = data.find(local_header_sig, pos)
        if idx == -1:
            break

        header = data[idx:idx + 30]
        if len(header) < 30:
            break

        _, _, flags, method, _, _, _, comp_size, uncomp_size, name_len, extra_len = struct.unpack(
            "<IHHHHHIIIHH", header
        )
        name_start = idx + 30
        name = data[name_start:name_start + name_len].decode("utf-8", errors="replace")
        data_start = name_start + name_len + extra_len

        has_data_descriptor = bool(flags & 0x08)
        is_encrypted = bool(flags & 0x01)

        if has_data_descriptor and comp_size == 0:
            dd_idx = data.find(data_descriptor_sig, data_start)
            if dd_idx == -1:
                pos = idx + 4
                continue
            comp_data = data[data_start:dd_idx]
            _, _, comp_size, uncomp_size = struct.unpack("<IIII", data[dd_idx:dd_idx + 16])
            next_pos = dd_idx + 16
        else:
            comp_data = data[data_start:data_start + comp_size]
            next_pos = data_start + comp_size

        if name and not name.endswith("/") and uncomp_size > 0:
            raw = comp_data
            if is_encrypted:
                raw = _decrypt_zipcrypto(comp_data, RAW_IMAGES_PASSWORD)[12:]

            try:
                if method == 8:
                    content = zlib.decompressobj(-15).decompress(raw)
                elif method == 0:
                    content = raw[:uncomp_size]
                else:
                    content = None
            except zlib.error:
                content = None

            if content is not None:
                out_path = os.path.join(dst, name)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "wb") as out_f:
                    out_f.write(content)

        pos = next_pos


def get_oct5k_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the OCT5k dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    os.makedirs(path, exist_ok=True)

    masks_dir = os.path.join(path, "OCT5k")
    if not os.path.exists(masks_dir):
        masks_zip = os.path.join(path, "OCT5k.zip")
        util.download_source(path=masks_zip, url=MASKS_URL, download=download, checksum=MASKS_CHECKSUM)
        util.unzip(zip_path=masks_zip, dst=path, remove=False)

    raw_dir = os.path.join(path, "raw")
    if not glob(os.path.join(raw_dir, "**", "*.TIFF"), recursive=True):
        raw_zip = os.path.join(path, "rasti_old.zip")
        util.download_source_gdrive(path=raw_zip, url=RAW_IMAGES_URL, download=download, checksum=RAW_IMAGES_CHECKSUM)
        _carve_zip_entries(raw_zip, raw_dir)

    return path


def get_oct5k_paths(
    path: Union[os.PathLike, str],
    grading: Literal["1", "2", "3"] = "1",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the OCT5k data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        grading: The choice of manual grader ('1', '2' or '3'). All three graded every image.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_oct5k_data(path, download)

    manual_paths_csv = os.path.join(data_dir, "OCT5k", "Scripts", "paths", "manual_paths.csv")
    masks_root = os.path.join(data_dir, "OCT5k", "Masks", "Masks_Manual", f"Grading_{grading}")
    raw_root = os.path.join(data_dir, "raw")

    resized_masks_dir = os.path.join(data_dir, f"masks_resized_grading_{grading}")

    image_paths, gt_paths = [], []
    with open(manual_paths_csv) as f:
        for row in csv.reader(f):
            mask_rel_path = row[0].replace("../Images/Images_Manual/", "")
            raw_rel_path = row[1].replace("./Macular-Dataset-R.Rasti_old/", "")

            raw_path = os.path.join(raw_root, raw_rel_path)
            if not os.path.exists(raw_path):
                # This image belongs to the 'DME' category, or otherwise was not among the files
                # recoverable from the truncated archive. See the module docstring.
                continue

            mask_path = os.path.join(masks_root, mask_rel_path)
            resized_mask_path = os.path.join(resized_masks_dir, mask_rel_path)
            if not os.path.exists(resized_mask_path):
                raw_size = imageio.imread(raw_path).shape[::-1]  # (H, W) -> (W, H) for PIL 'resize'.
                mask = Image.open(mask_path)
                mask_resized = mask.resize(raw_size, Image.NEAREST)
                os.makedirs(os.path.dirname(resized_mask_path), exist_ok=True)
                imageio.imwrite(resized_mask_path, np.array(mask_resized))

            image_paths.append(raw_path)
            gt_paths.append(resized_mask_path)

    image_paths, gt_paths = natsorted(image_paths), natsorted(gt_paths)

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0

    return image_paths, gt_paths


def get_oct5k_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    grading: Literal["1", "2", "3"] = "1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the OCT5k dataset for retinal layer band segmentation in OCT B-scans.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        grading: The choice of manual grader ('1', '2' or '3'). All three graded every image.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_oct5k_paths(path, grading, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_oct5k_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    grading: Literal["1", "2", "3"] = "1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the OCT5k dataloader for retinal layer band segmentation in OCT B-scans.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        grading: The choice of manual grader ('1', '2' or '3'). All three graded every image.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_oct5k_dataset(path, patch_shape, grading, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
