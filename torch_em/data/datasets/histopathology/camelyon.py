"""This dataset contains annotations for tumor region segmentation in
whole-slide histopathology images of breast cancer sentinel lymph node sections.

The data is from the CAMELYON16 and CAMELYON17 challenges, described in
https://doi.org/10.1093/gigascience/giy065 ("1399 H&E-stained sentinel lymph
node sections of breast cancer patients: the CAMELYON dataset"). It is hosted
on the public AWS Open Data Registry (https://registry.opendata.aws/camelyon/)
under a CC0 license. Please cite the publication if you use this dataset.

Two segmentation-compatible sources are exposed:
- CAMELYON16: all 399 whole-slide images (159 normal, 111 tumor, 129 test),
  each paired with a tumor segmentation mask.
- CAMELYON17: the subset of 100 training whole-slide images that ship with a
  segmentation mask. The remaining CAMELYON17 slides only carry a patient-level
  pN-stage classification label and are out of scope for this loader.

Mask label values: 0 (background / non-tumor tissue), 1 (tumor), and for
CAMELYON16 tumor slides additionally 2 (non-tumor tissue excluded from a
tumor annotation).

NOTE: The whole-slide images and masks are multi-resolution pyramidal TIFFs of
several gigabytes each. On the first use each requested slide is converted into
a chunked HDF5 file at the requested pyramid level, which requires some time
and disk space.
"""

import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://camelyon-dataset.s3.us-west-2.amazonaws.com"

CHECKSUM_URLS = {
    "CAMELYON16": f"{BASE_URL}/CAMELYON16/checksums.md5",
    "CAMELYON17": f"{BASE_URL}/CAMELYON17/checksums.md5",
}


def _get_checksums(path, version, download):
    checksum_path = os.path.join(path, f"{version.lower()}_checksums.md5")
    util.download_source(path=checksum_path, url=CHECKSUM_URLS[version], download=download, checksum=None)

    checksums = {}
    with open(checksum_path) as f:
        for line in f:
            checksum, name = line.split(maxsplit=1)
            checksums[name.strip().lstrip("*")] = checksum
    return checksums


def _verify_md5(path, expected):
    import hashlib

    actual = hashlib.md5(Path(path).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"The checksum of '{path}' does not match the expected checksum: {expected} != {actual}")


def _download_slide(path, version, stem, checksums, download):
    raw_dir = os.path.join(path, "raw", version)
    os.makedirs(raw_dir, exist_ok=True)

    image_path = os.path.join(raw_dir, f"{stem}.tif")
    mask_path = os.path.join(raw_dir, f"{stem}_mask.tif")

    for out_path, key in [(image_path, f"images/{stem}.tif"), (mask_path, f"masks/{stem}_mask.tif")]:
        if os.path.exists(out_path):
            continue
        util.download_source(path=out_path, url=f"{BASE_URL}/{version}/{key}", download=download, checksum=None)
        _verify_md5(out_path, checksums[key])

    return image_path, mask_path


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

    # Some scanners (e.g. Philips) round the declared level shape slightly differently between the raw
    # slide and the independently rasterized mask, so tolerate a small mismatch and crop to the overlap.
    image_height, image_width = image_series.levels[resolution_level].shape[:2]
    mask_height, mask_width = mask_series.levels[resolution_level].shape[:2]
    height, width = min(image_height, mask_height), min(image_width, mask_width)
    tolerance = 0.02
    if abs(image_height - mask_height) > tolerance * image_height or \
            abs(image_width - mask_width) > tolerance * image_width:
        raise RuntimeError(
            f"The mask '{mask_path}' does not match the raw shape ({image_height}, {image_width}) "
            f"at level {resolution_level}: got ({mask_height}, {mask_width})."
        )

    image = _open_level(image_series, resolution_level)
    mask = _open_level(mask_series, resolution_level)

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


def _restrict_to_sample_ids(stems, sample_ids):
    if sample_ids is None:
        return stems
    missing = sorted(set(sample_ids) - set(stems))
    if missing:
        raise ValueError(f"The following sample ids are not part of this dataset: {missing}")
    return [stem for stem in stems if stem in sample_ids]


def _resolve_camelyon16_stems(checksums, split, sample_ids):
    stems = sorted(Path(name).stem for name in checksums if name.startswith("images/"))
    if split is not None:
        assert split in ("train", "test"), "Please choose from the available `train` / `test` splits"
        prefixes = ("normal_", "tumor_") if split == "train" else ("test_",)
        stems = [stem for stem in stems if stem.startswith(prefixes)]
    return _restrict_to_sample_ids(stems, sample_ids)


def _resolve_camelyon17_stems(checksums, sample_ids):
    stems = sorted(Path(name).stem[:-len("_mask")] for name in checksums if name.startswith("masks/"))
    return _restrict_to_sample_ids(stems, sample_ids)


def get_camelyon16_data(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "test"]] = None,
    sample_ids: Optional[List[str]] = None,
    resolution_level: int = 0,
    download: bool = False,
) -> str:
    """Download and preprocess the CAMELYON16 tumor segmentation data.

    Args:
        path: Filepath to a folder where the data will be saved.
        split: The split to use. Either 'train' (normal and tumor slides) or 'test'. By default all slides are used.
        sample_ids: The slide names to restrict the data to, e.g. ['tumor_091', 'normal_108'].
            By default all slides (matching `split`) are used.
        resolution_level: The pyramid level to convert. 0 is the native resolution; use a higher level to
            reduce the size of the preprocessed data.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the preprocessed data is stored.
    """
    checksums = _get_checksums(path, "CAMELYON16", download)
    stems = _resolve_camelyon16_stems(checksums, split, sample_ids)

    preprocessed_dir = os.path.join(path, "preprocessed", "CAMELYON16")
    os.makedirs(preprocessed_dir, exist_ok=True)

    for stem in stems:
        output_path = os.path.join(preprocessed_dir, f"{stem}_level{resolution_level}.h5")
        if os.path.exists(output_path):
            continue
        image_path, mask_path = _download_slide(path, "CAMELYON16", stem, checksums, download)
        _convert_slide(image_path, mask_path, output_path, resolution_level)

    return preprocessed_dir


def get_camelyon16_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "test"]] = None,
    sample_ids: Optional[List[str]] = None,
    resolution_level: int = 0,
    download: bool = False,
) -> List[str]:
    """Get paths to the CAMELYON16 tumor segmentation data.

    Args:
        path: Filepath to a folder where the data will be saved.
        split: The split to use. Either 'train' (normal and tumor slides) or 'test'. By default all slides are used.
        sample_ids: The slide names to restrict the data to, e.g. ['tumor_091', 'normal_108'].
            By default all slides (matching `split`) are used.
        resolution_level: The pyramid level to convert. 0 is the native resolution; use a higher level to
            reduce the size of the preprocessed data.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the preprocessed HDF5 files.
    """
    preprocessed_dir = get_camelyon16_data(path, split, sample_ids, resolution_level, download)
    stems = _resolve_camelyon16_stems(_get_checksums(path, "CAMELYON16", download=False), split, sample_ids)
    return [os.path.join(preprocessed_dir, f"{stem}_level{resolution_level}.h5") for stem in stems]


def get_camelyon17_data(
    path: Union[os.PathLike, str],
    sample_ids: Optional[List[str]] = None,
    resolution_level: int = 0,
    download: bool = False,
) -> str:
    """Download and preprocess the CAMELYON17 tumor segmentation subset.

    This is the subset of CAMELYON17 training slides that ship with a segmentation mask.
    The remaining slides only carry a patient-level classification label and are not included.

    Args:
        path: Filepath to a folder where the data will be saved.
        sample_ids: The slide names to restrict the data to, e.g. ['patient_000_node_4'].
            By default all slides with a segmentation mask are used.
        resolution_level: The pyramid level to convert. 0 is the native resolution; use a higher level to
            reduce the size of the preprocessed data.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the preprocessed data is stored.
    """
    checksums = _get_checksums(path, "CAMELYON17", download)
    stems = _resolve_camelyon17_stems(checksums, sample_ids)

    preprocessed_dir = os.path.join(path, "preprocessed", "CAMELYON17")
    os.makedirs(preprocessed_dir, exist_ok=True)

    for stem in stems:
        output_path = os.path.join(preprocessed_dir, f"{stem}_level{resolution_level}.h5")
        if os.path.exists(output_path):
            continue
        image_path, mask_path = _download_slide(path, "CAMELYON17", stem, checksums, download)
        _convert_slide(image_path, mask_path, output_path, resolution_level)

    return preprocessed_dir


def get_camelyon17_paths(
    path: Union[os.PathLike, str],
    sample_ids: Optional[List[str]] = None,
    resolution_level: int = 0,
    download: bool = False,
) -> List[str]:
    """Get paths to the CAMELYON17 tumor segmentation subset.

    Args:
        path: Filepath to a folder where the data will be saved.
        sample_ids: The slide names to restrict the data to, e.g. ['patient_000_node_4'].
            By default all slides with a segmentation mask are used.
        resolution_level: The pyramid level to convert. 0 is the native resolution; use a higher level to
            reduce the size of the preprocessed data.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the preprocessed HDF5 files.
    """
    preprocessed_dir = get_camelyon17_data(path, sample_ids, resolution_level, download)
    stems = _resolve_camelyon17_stems(_get_checksums(path, "CAMELYON17", download=False), sample_ids)
    return [os.path.join(preprocessed_dir, f"{stem}_level{resolution_level}.h5") for stem in stems]


def _get_dataset(
    volume_paths, patch_shape, label_dtype, resize_inputs, kwargs,
) -> Dataset:
    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

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


def get_camelyon16_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "test"]] = None,
    sample_ids: Optional[List[str]] = None,
    resolution_level: int = 0,
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> Dataset:
    """Get the CAMELYON16 dataset for tumor segmentation in whole-slide histopathology images.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape to use for training.
        split: The split to use. Either 'train' (normal and tumor slides) or 'test'. By default all slides are used.
        sample_ids: The slide names to restrict the data to, e.g. ['tumor_091', 'normal_108'].
            By default all slides (matching `split`) are used.
        resolution_level: The pyramid level to convert. 0 is the native resolution; use a higher level to
            reduce the size of the preprocessed data.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of the labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_camelyon16_paths(path, split, sample_ids, resolution_level, download)
    return _get_dataset(volume_paths, patch_shape, label_dtype, resize_inputs, kwargs)


def get_camelyon16_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: Optional[Literal["train", "test"]] = None,
    sample_ids: Optional[List[str]] = None,
    resolution_level: int = 0,
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CAMELYON16 dataloader for tumor segmentation in whole-slide histopathology images.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        split: The split to use. Either 'train' (normal and tumor slides) or 'test'. By default all slides are used.
        sample_ids: The slide names to restrict the data to, e.g. ['tumor_091', 'normal_108'].
            By default all slides (matching `split`) are used.
        resolution_level: The pyramid level to convert. 0 is the native resolution; use a higher level to
            reduce the size of the preprocessed data.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of the labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_camelyon16_dataset(
        path=path, patch_shape=patch_shape, split=split, sample_ids=sample_ids, resolution_level=resolution_level,
        download=download, label_dtype=label_dtype, resize_inputs=resize_inputs, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)


def get_camelyon17_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    sample_ids: Optional[List[str]] = None,
    resolution_level: int = 0,
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> Dataset:
    """Get the CAMELYON17 tumor segmentation subset for whole-slide histopathology images.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape to use for training.
        sample_ids: The slide names to restrict the data to, e.g. ['patient_000_node_4'].
            By default all slides with a segmentation mask are used.
        resolution_level: The pyramid level to convert. 0 is the native resolution; use a higher level to
            reduce the size of the preprocessed data.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of the labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_camelyon17_paths(path, sample_ids, resolution_level, download)
    return _get_dataset(volume_paths, patch_shape, label_dtype, resize_inputs, kwargs)


def get_camelyon17_loader(
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
    """Get the CAMELYON17 tumor segmentation subset dataloader for whole-slide histopathology images.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        sample_ids: The slide names to restrict the data to, e.g. ['patient_000_node_4'].
            By default all slides with a segmentation mask are used.
        resolution_level: The pyramid level to convert. 0 is the native resolution; use a higher level to
            reduce the size of the preprocessed data.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of the labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_camelyon17_dataset(
        path=path, patch_shape=patch_shape, sample_ids=sample_ids, resolution_level=resolution_level,
        download=download, label_dtype=label_dtype, resize_inputs=resize_inputs, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
