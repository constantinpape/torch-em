"""The Parlakgul fasting liver dataset contains FIB-SEM volumes of mouse liver in fasting and obesity
with dense organelle segmentations.

Four FIB-SEM volumes (8 nm isotropic) are available:
- 9428 (lean fasted): 7984 x 6984 x 9501 voxels
- 8970 (obese fasted): 9402 x 6949 x 8450 voxels
- 2679 (obese, LacZ control): 6500 x 6500 x 7705 voxels
- 1858 (obese, RRBP1 overexpression): 9375 x 10625 x 5795 voxels (the first 65 slices contain resin and are removed)

Segmentations are available via the `label_choice` parameter:
- "er": endoplasmic reticulum, binary (all samples)
- "mito": mitochondria, instance segmentation (all samples)
- "lipid_droplet": lipid droplets, binary (9428 and 8970)
- "nuclear_membrane": nuclear membrane, binary (9428 and 8970)
- "cells": cells, instance segmentation (9428 and 8970). Only a subset of the cells is annotated, the remaining
  cells are background, so these labels are not suitable for dense cell segmentation.

The binary labels are converted to 0 (background) and 1 (foreground). The instance ids are made consecutive
within each requested subvolume. Note that the labels of a large bounding box are held in memory while streaming
(4 bytes per voxel for "mito").

Data is streamed lazily from EMPIAR-12017 via HTTP range requests: only the rows of the raw TIFFs and the
segmentation slices (from the ZIP archives) of the requested bounding box are fetched and cached as zarr v3.

Bounding boxes are specified as (x_min, x_max, y_min, y_max, z_min, z_max) in voxels.

This dataset is from the publication https://doi.org/10.1038/s41467-024-48272-7.
Please cite it if you use this dataset in your research.

The data is publicly available at https://www.ebi.ac.uk/empiar/EMPIAR-12017/.
"""

import os
import shutil
import zipfile
from concurrent import futures
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch_em
from .. import util
from .parlakgul_liver import (
    _HttpFile, _bbox_to_str, _index_zip_slices, _read_raw_slice, _read_zip_slice, N_THREADS
)


EMPIAR_BASE = "https://ftp.ebi.ac.uk/empiar/world_availability/12017/data"

# The raw file of slice z is 'raw_pattern' formatted with z + 'raw_z_offset'. The first 'z_offset' slices of
# a sample are removed (they contain resin), so the shape and all slice indices are relative to this offset.
# The segmentation slices are found by the slice number in their name, see '_index_zip_slices'.
PARLAKGUL_FASTING_SAMPLES: Dict[str, dict] = {
    "9428": {
        "condition": "lean_fasted",
        "dir": "9428%20Lean%20Fasted",
        "raw_dir": "Lean_Fasted_9428_raw_8-bit",
        "raw_pattern": "Lean_Fasted_9428_raw_T001_Z{z:03d}_C05.tif",
        "z_offset": 0,
        "shape": (9501, 6984, 7984),
        "seg_zips": {
            "er": "Lean_Fasted_9428_ER_8-bit.zip",
            "mito": "Lean_Fasted_9428_mitochondria_instance_32-bit.zip",
            "lipid_droplet": "Lean_Fasted_9428_LD_8-bit.zip",
            "nuclear_membrane": "Lean_Fasted_9428_nucleus_8-bit.zip",
            "cells": "Lean_Fasted_9428_plasma_membrane_8-bit.zip",
        },
    },
    "8970": {
        "condition": "obese_fasted",
        "dir": "8970%20Obese%20Fasted",
        "raw_dir": "Obese_Fasted_8970_raw_8-bit",
        "raw_pattern": "Obese_Fasted_8970_raw_T001_Z{z:03d}_C05.tif",
        "z_offset": 0,
        "shape": (8450, 6949, 9402),
        "seg_zips": {
            "er": "Obese_Fasted_8970_ER_8-bit.zip",
            "mito": "Obese_fasted_8970_mitochondria_instance_32-bit.zip",
            "lipid_droplet": "Obese_Fasted_8970_LD_8-bit.zip",
            "nuclear_membrane": "Obese_Fasted_8970_nucleus_8-bit.zip",
            "cells": "Obese_fasted_8970_plasma_membrane_8-bit.zip",
        },
    },
    "2679": {
        "condition": "obese_lacz",
        "dir": "Obese%20LacZ",
        "raw_dir": "ob_LacZ_2679_raw_8-bit",
        "raw_pattern": "2679_Ob_LacZ_T001_Z{z:03d}_C01.tif",
        "z_offset": 0,
        "shape": (7705, 6500, 6500),
        "seg_zips": {
            "er": "ob_LacZ_2679_ER_8-bit.zip",
            "mito": "ob_LacZ_2679_mitochondria-instance_32-bit.zip",
        },
    },
    "1858": {
        "condition": "obese_rrbp1",
        "dir": "Obese%20RRBP1",
        "raw_dir": "1858_Ob_RRBP1-OE_Raw_8-bit",
        "raw_pattern": "1858_Ob_RRBP1-OE_Raw_T001_Z{z:03d}_C01.tif",
        "z_offset": 65,
        "shape": (5795, 10625, 9375),
        "seg_zips": {
            "er": "ob_RRBP1_OE_1858_ER_8-bit.zip",
            "mito": "ob_RRBP1_OE_1858_mito-instance_32-bit.zip",
        },
    },
}

INSTANCE_LABELS = {"mito": np.uint32, "cells": np.uint8}
CHUNK_SHAPE = (64, 256, 256)
LabelChoice = Literal["er", "mito", "lipid_droplet", "nuclear_membrane", "cells"]
SampleChoice = Literal["9428", "8970", "2679", "1858"]


def get_parlakgul_fasting_liver_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[int, int, int, int, int, int],
    sample: SampleChoice = "9428",
    label_choice: LabelChoice = "mito",
    download: bool = False,
) -> str:
    """Stream a subvolume from the Parlakgul fasting liver dataset and cache it as a zarr v3 store.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        bounding_box: The region to fetch as (x_min, x_max, y_min, y_max, z_min, z_max)
            in voxel coordinates at 8 nm isotropic resolution.
        sample: Which liver sample to use. One of "9428" (lean fasted), "8970" (obese fasted),
            "2679" (obese LacZ) or "1858" (obese RRBP1).
        label_choice: Which segmentation to use as labels. One of "er", "mito", "lipid_droplet",
            "nuclear_membrane" or "cells". The last three are only available for "9428" and "8970".
        download: Whether to stream and cache the data if it is not present.

    Returns:
        The filepath to the cached zarr store.
    """
    import zarr
    from skimage.segmentation import relabel_sequential
    from zarr.codecs import BloscCodec

    if sample not in PARLAKGUL_FASTING_SAMPLES:
        raise ValueError(f"'{sample}' is not a valid sample. Choose from {list(PARLAKGUL_FASTING_SAMPLES)}.")
    sample_info = PARLAKGUL_FASTING_SAMPLES[sample]
    if label_choice not in sample_info["seg_zips"]:
        raise ValueError(
            f"label_choice='{label_choice}' is not available for sample='{sample}'. "
            f"Choose from {list(sample_info['seg_zips'])}."
        )

    x_min, x_max, y_min, y_max, z_min, z_max = bounding_box
    n_z, n_y, n_x = sample_info["shape"]
    if not (0 <= x_min < x_max <= n_x and 0 <= y_min < y_max <= n_y and 0 <= z_min < z_max <= n_z):
        raise ValueError(f"The bounding box {bounding_box} is not inside sample '{sample}' of shape (z, y, x) "
                         f"{sample_info['shape']}.")

    os.makedirs(str(path), exist_ok=True)
    zarr_path = os.path.join(str(path), f"{sample}_{label_choice}_{_bbox_to_str(bounding_box)}.zarr")
    if os.path.exists(os.path.join(zarr_path, "raw")) and os.path.exists(os.path.join(zarr_path, "labels")):
        return zarr_path

    if not download:
        raise RuntimeError(
            f"No cached data found at '{zarr_path}'. Set download=True to stream it from EMPIAR."
        )

    shape = (z_max - z_min, y_max - y_min, x_max - x_min)
    label_dtype = INSTANCE_LABELS.get(label_choice, np.uint8)
    raw_arr = np.zeros(shape, dtype=np.uint8)
    lbl_arr = np.zeros(shape, dtype=label_dtype)

    sample_base = f"{EMPIAR_BASE}/{sample_info['dir']}"
    seg_zip_url = f"{sample_base}/{sample_info['seg_zips'][label_choice]}"
    z_offset = sample_info["z_offset"]

    print(f"Streaming Parlakgul fasting liver {sample} ({sample_info['condition']}) EM + {label_choice} ...")

    # The ZIP directory is read once, the segmentation slices are then fetched one by one from it.
    zf = zipfile.ZipFile(_HttpFile(seg_zip_url))
    slices = _index_zip_slices(zf)
    if len(slices) != n_z + z_offset:
        raise RuntimeError(f"Expected {n_z + z_offset} segmentation slices in {seg_zip_url}, found {len(slices)}.")

    def _fetch_raw(i):
        # The raw files are numbered from 1.
        name = sample_info["raw_pattern"].format(z=z_min + i + z_offset + 1)
        raw_arr[i] = _read_raw_slice(f"{sample_base}/{sample_info['raw_dir']}/{name}", x_min, x_max, y_min, y_max)

    with futures.ThreadPoolExecutor(N_THREADS) as tp:
        raw_tasks = [tp.submit(_fetch_raw, i) for i in range(shape[0])]
        for i, z in enumerate(range(z_min, z_max)):
            seg = _read_zip_slice(zf, slices[z + z_offset], x_min, x_max, y_min, y_max)
            lbl_arr[i] = seg if label_choice in INSTANCE_LABELS else seg > 0
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{shape[0]} segmentation slices done")
        for task in raw_tasks:
            task.result()

    # The instance ids are global to the whole volume, so they are made consecutive within the subvolume.
    if label_choice in INSTANCE_LABELS:
        lbl_arr = relabel_sequential(lbl_arr)[0].astype(label_dtype)

    # The store is written under a temporary name and moved in place once it is complete,
    # so that an interrupted download is not mistaken for a complete one.
    tmp_path = f"{zarr_path}.incomplete"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    root = zarr.open_group(tmp_path, mode="w")
    root.attrs.update({
        "bounding_box": list(bounding_box), "sample": sample, "label_choice": label_choice, "resolution_nm": [8, 8, 8],
    })
    for name, data, shuffle in (("raw", raw_arr, "shuffle"), ("labels", lbl_arr, "bitshuffle")):
        arr = root.create_array(
            name, shape=data.shape, chunks=CHUNK_SHAPE, dtype=data.dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        arr[:] = data
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    os.rename(tmp_path, zarr_path)

    print(f"Cached to {zarr_path} (shape {shape})")
    return zarr_path


def get_parlakgul_fasting_liver_paths(
    path: Union[os.PathLike, str],
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    sample: SampleChoice = "9428",
    label_choice: LabelChoice = "mito",
    download: bool = False,
) -> List[str]:
    """Get paths to Parlakgul fasting liver zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        bounding_boxes: List of regions to fetch, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in voxel coordinates.
        sample: Which liver sample to use.
        label_choice: Which segmentation to use as labels.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    return [get_parlakgul_fasting_liver_data(path, bbox, sample, label_choice, download) for bbox in bounding_boxes]


def get_parlakgul_fasting_liver_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    sample: SampleChoice = "9428",
    label_choice: LabelChoice = "mito",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Parlakgul fasting liver dataset for organelle segmentation.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel coordinates.
        sample: Which liver sample to use. One of "9428", "8970", "2679", "1858".
        label_choice: Which segmentation to use. One of "er", "mito", "lipid_droplet",
            "nuclear_membrane" or "cells".
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_parlakgul_fasting_liver_paths(path, bounding_boxes, sample, label_choice, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_parlakgul_fasting_liver_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    sample: SampleChoice = "9428",
    label_choice: LabelChoice = "mito",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for organelle segmentation in the Parlakgul fasting liver dataset.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel coordinates.
        sample: Which liver sample to use. One of "9428", "8970", "2679", "1858".
        label_choice: Which segmentation to use. One of "er", "mito", "lipid_droplet",
            "nuclear_membrane" or "cells".
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_parlakgul_fasting_liver_dataset(
        path, patch_shape, bounding_boxes, sample=sample, label_choice=label_choice,
        download=download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
