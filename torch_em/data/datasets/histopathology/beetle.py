"""BEETLE (BrEast cancEr hisTopathoLogy sEgmentation) contains multiclass semantic
segmentation annotations for H&E stained breast cancer whole-slide images (WSIs),
collected across multiple clinical centers and scanners.

Mask label values (see the dataset's 'label_map.json'): 0 (unannotated), 1 (other),
2 (non-invasive epithelium), 3 (invasive epithelium), 4 (necrosis).

NOTE: Only the 'development' set is exposed here. Its own official split uses 5
cross-validation folds; pass `validation_fold` to choose which fold is held out for
validation. The dataset also has an 'evaluation' set, but its annotations are not
publicly released (they are sequestered on the Grand Challenge platform), so it is
not included. A further subset of development slides sourced from TCGA is not bundled
as raw images in this Zenodo deposit (their raw whole-slide images would need to be
fetched separately from the NCI Genomic Data Commons) and is excluded as well.

NOTE: The whole-slide images and masks are multi-resolution pyramidal TIFFs bundled
inside two large Zenodo archives (images: about 147 GB; masks: about 1.8 GB). Requested
slides are extracted directly from these remote archives without downloading the full
archives, then converted into a chunked HDF5 file at the requested pyramid level.

This dataset is located at https://doi.org/10.5281/zenodo.16812932.
This dataset is from the publication https://arxiv.org/abs/2510.02037.
Please cite it if you use this dataset for your research.
"""

import os
import csv
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


OVERVIEW_URL = "https://zenodo.org/api/records/16812932/files/data_overview.csv/content"
OVERVIEW_CHECKSUM = "a063a5456959cb92f3dc007844cb14208a71336d8157d006304539066d86f81b"

IMAGES_ZIP_URL = "https://zenodo.org/api/records/16812932/files/images.zip/content"
ANNOTATIONS_ZIP_URL = "https://zenodo.org/api/records/16812932/files/annotations.zip/content"


def _load_manifest(path, download):
    csv_path = os.path.join(path, "data_overview.csv")
    util.download_source(path=csv_path, url=OVERVIEW_URL, download=download, checksum=OVERVIEW_CHECKSUM)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    # Only the development set ships raw whole-slide images in this deposit; the rest
    # (evaluation set, and a subset of development rows sourced from TCGA) are excluded.
    return [row for row in rows if row["split"] == "development" and row["wsi_path"]]


def _resolve_rows(path, split, validation_fold, sample_ids, download):
    rows = _load_manifest(path, download)
    fold = f"fold{validation_fold}"
    rows = [row for row in rows if (row["validation_fold"] == fold) == (split == "val")]
    if sample_ids is not None:
        by_name = {row["name"]: row for row in rows}
        missing = sorted(set(sample_ids) - set(by_name))
        if missing:
            raise ValueError(f"The following sample ids are not part of this split: {missing}")
        rows = [by_name[name] for name in sample_ids]
    return rows


def _extract_zip_member(zip_url, member, out_path):
    import fsspec
    import zipfile

    if os.path.exists(out_path):
        return
    fs = fsspec.filesystem("http")
    with fs.open(zip_url) as f:
        data = zipfile.ZipFile(f).read(member)
    tmp_path = out_path + ".tmp"
    Path(tmp_path).write_bytes(data)
    os.replace(tmp_path, out_path)


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
    height, width = mask_series.levels[resolution_level].shape[:2]

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


def get_beetle_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "val"],
    validation_fold: int = 0,
    sample_ids: Optional[List[str]] = None,
    resolution_level: int = 0,
    download: bool = False,
) -> str:
    """Download and preprocess the BEETLE breast cancer segmentation data.

    Args:
        path: Filepath to a folder where the data will be saved.
        split: The split to use, either the held-out validation fold or the rest.
        validation_fold: Which of the 5 official cross-validation folds (0-4) to use as validation.
        sample_ids: The slide names to restrict the data to, e.g. ['patient1_wsi1'].
            By default all slides matching `split` and `validation_fold` are used.
        resolution_level: The pyramid level to convert. 0 is the native resolution; use a higher level to
            reduce the size of the preprocessed data.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the preprocessed data is stored.
    """
    rows = _resolve_rows(path, split, validation_fold, sample_ids, download)

    raw_dir = os.path.join(path, "raw")
    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)

    for row in rows:
        output_path = os.path.join(preprocessed_dir, f"{row['name']}_level{resolution_level}.h5")
        if os.path.exists(output_path):
            continue
        if not download:
            raise RuntimeError(f"Cannot find the data at {output_path}, but download was set to False")
        image_path = os.path.join(raw_dir, f"{row['name']}.tif")
        mask_path = os.path.join(raw_dir, f"{row['name']}_mask.tif")
        _extract_zip_member(IMAGES_ZIP_URL, row["wsi_path"], image_path)
        _extract_zip_member(ANNOTATIONS_ZIP_URL, row["annotation_mask_path"], mask_path)
        _convert_slide(image_path, mask_path, output_path, resolution_level)

    return preprocessed_dir


def get_beetle_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val"],
    validation_fold: int = 0,
    sample_ids: Optional[List[str]] = None,
    resolution_level: int = 0,
    download: bool = False,
) -> List[str]:
    """Get paths to the BEETLE breast cancer segmentation data.

    Args:
        path: Filepath to a folder where the data will be saved.
        split: The split to use, either the held-out validation fold or the rest.
        validation_fold: Which of the 5 official cross-validation folds (0-4) to use as validation.
        sample_ids: The slide names to restrict the data to, e.g. ['patient1_wsi1'].
            By default all slides matching `split` and `validation_fold` are used.
        resolution_level: The pyramid level to convert. 0 is the native resolution; use a higher level to
            reduce the size of the preprocessed data.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the preprocessed HDF5 files.
    """
    preprocessed_dir = get_beetle_data(path, split, validation_fold, sample_ids, resolution_level, download)
    rows = _resolve_rows(path, split, validation_fold, sample_ids, download=False)
    return [os.path.join(preprocessed_dir, f"{row['name']}_level{resolution_level}.h5") for row in rows]


def get_beetle_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"],
    validation_fold: int = 0,
    sample_ids: Optional[List[str]] = None,
    resolution_level: int = 0,
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> Dataset:
    """Get the BEETLE dataset for multiclass breast cancer tissue segmentation in whole-slide images.

    Args:
        path: Filepath to a folder where the data will be saved.
        patch_shape: The patch shape to use for training.
        split: The split to use, either the held-out validation fold or the rest.
        validation_fold: Which of the 5 official cross-validation folds (0-4) to use as validation.
        sample_ids: The slide names to restrict the data to, e.g. ['patient1_wsi1'].
            By default all slides matching `split` and `validation_fold` are used.
        resolution_level: The pyramid level to convert. 0 is the native resolution; use a higher level to
            reduce the size of the preprocessed data.
        download: Whether to download the data if it is not present.
        label_dtype: The datatype of the labels.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_beetle_paths(path, split, validation_fold, sample_ids, resolution_level, download)

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


def get_beetle_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"],
    validation_fold: int = 0,
    sample_ids: Optional[List[str]] = None,
    resolution_level: int = 0,
    download: bool = False,
    label_dtype: torch.dtype = torch.int64,
    resize_inputs: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BEETLE dataloader for multiclass breast cancer tissue segmentation in whole-slide images.

    Args:
        path: Filepath to a folder where the data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The split to use, either the held-out validation fold or the rest.
        validation_fold: Which of the 5 official cross-validation folds (0-4) to use as validation.
        sample_ids: The slide names to restrict the data to, e.g. ['patient1_wsi1'].
            By default all slides matching `split` and `validation_fold` are used.
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
    dataset = get_beetle_dataset(
        path=path, patch_shape=patch_shape, split=split, validation_fold=validation_fold, sample_ids=sample_ids,
        resolution_level=resolution_level, download=download, label_dtype=label_dtype, resize_inputs=resize_inputs,
        **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
