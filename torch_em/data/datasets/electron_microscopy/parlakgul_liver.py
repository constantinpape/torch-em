"""The Parlakgul liver dataset contains FIB-SEM volumes of mouse liver with dense
semantic segmentation of 7 organelle classes. All labels are binary semantic masks
(0=background, 1=foreground) - not instance segmentation.

Four FIB-SEM volumes are available across lean and obese conditions:
- 6461 (lean): 12000 x 8000 x 5638 voxels, 8 nm isotropic
- 6464 (obese 1): 9112 x 10200 x 7896 voxels, 8 nm isotropic
- 9430 (obese 2): 8000 x 8050 x 8501 voxels, 8 nm isotropic
- 1857 (obese Climp63): 9700 x 9650 x 3629 voxels, 8 nm isotropic

Seven semantic segmentation classes are available via the `label_choice` parameter:
- "er": endoplasmic reticulum
- "er_sheets": ER sheets
- "er_tubules": ER tubules
- "mito": mitochondria
- "lipid_droplet": lipid droplets
- "nuclear_membrane": nuclear membrane
- "plasma_membrane": plasma membrane (not available for 1857)

Data is streamed lazily from EMPIAR-10791 via HTTP: raw TIFFs are fetched per z-slice,
segmentation is extracted per z-slice from ZIP archives using HTTP range requests.
Only the requested bounding box region is downloaded and cached as zarr v3.

Bounding boxes are specified as (x_min, x_max, y_min, y_max, z_min, z_max) in voxels.

This dataset is from the publication https://doi.org/10.1038/s41586-022-04518-2.
Please cite it if you use this dataset in your research.

The data is publicly available at https://www.ebi.ac.uk/empiar/EMPIAR-10791/.
"""

import hashlib
import io
import os
import zipfile
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch_em
from .. import util


EMPIAR_BASE = "https://ftp.ebi.ac.uk/empiar/world_availability/10791/data"
PARLAKGUL_PAPER_BASE = (
    "Parlakgul%20et%20al%20-%20Regulation%20of%20liver%20subcellular%20architecture%20"
    "controls%20metabolic%20homeostasis/FIB-SEM%20Raw%20and%20Segmentation%20Data"
)

PARLAKGUL_SAMPLES: Dict[str, dict] = {
    "6461": {
        "condition": "lean",
        "raw_dir": "6461%20-%20Lean%20Liver/6461%20Lean%20Liver%20-%20Raw",
        "seg_dir": "6461%20-%20Lean%20Liver/6461%20Lean%20Liver%20-%20Segmentation",
        "raw_pattern": "Gunes_WT1_8x8x8nm_3MHz.{z:04d}.tif",
        "shape": (5638, 8000, 12000),
        "seg_zips": {
            "er": "6461%20Lean%20ER.zip",
            "er_sheets": "6461%20Lean%20ER%20Sheets.zip",
            "er_tubules": "6461%20Lean%20ER%20Tubules.zip",
            "mito": "6461%20Lean%20Mitochondria.zip",
            "lipid_droplet": "6461%20Lean%20Lipid%20Droplet.zip",
            "nuclear_membrane": "6461%20Lean%20Nuclear%20membrane.zip",
            "plasma_membrane": "6461%20Lean%20Plasma%20Membrane.zip",
        },
    },
    "6464": {
        "condition": "obese1",
        "raw_dir": "6464%20-%20Obese1%20Liver/6464%20Obese1%20Liver%20-%20Raw",
        "seg_dir": "6464%20-%20Obese1%20Liver/6464%20Obese1%20Liver%20-%20Segmentation",
        "raw_pattern": "Gunes_HFD1_8x8x8nm_3MHz.{z:04d}.tif",
        "shape": (7896, 10200, 9112),
        "seg_zips": {
            "er": "6464%20Obese1%20ER.zip",
            "er_sheets": "6464%20Obese1%20ER%20Sheets.zip",
            "er_tubules": "6464%20Obese1%20ER%20Tubules.zip",
            "mito": "6464%20Obese1%20Mitochondria.zip",
            "lipid_droplet": "6464%20Obese1%20Lipid%20Droplet.zip",
            "nuclear_membrane": "6464%20Obese1%20Nuclear%20membrane.zip",
            "plasma_membrane": "6464%20Obese1%20Plasma%20Membrane.zip",
        },
    },
    "9430": {
        "condition": "obese2",
        "raw_dir": "9430%20-%20Obese2%20Liver/9430%20Obese2%20Liver%20-%20Raw",
        "seg_dir": "9430%20-%20Obese2%20Liver/9430%20Obese2%20Liver%20-%20Segmentation",
        "raw_pattern": "Gunes_HFD2_8x8x8nm_3MHz.{z:04d}.tif",
        "shape": (8501, 8050, 8000),
        "seg_zips": {
            "er": "9430%20Obese2%20ER.zip",
            "er_sheets": "9430%20Obese2%20ER%20Sheets.zip",
            "er_tubules": "9430%20Obese2%20ER%20Tubules.zip",
            "mito": "9430%20Obese2%20Mitochondria.zip",
            "lipid_droplet": "9430%20Obese2%20Lipid%20Droplet.zip",
            "nuclear_membrane": "9430%20Obese2%20Nuclear%20membrane.zip",
            "plasma_membrane": "9430%20Obese2%20Plasma%20Membrane.zip",
        },
    },
    "1857": {
        "condition": "obese_climp63",
        "raw_dir": "1857%20-%20Obese%20Climp-63%20Liver/1857%20Obese%20Climp63%20Liver%20-%20Raw",
        "seg_dir": "1857%20-%20Obese%20Climp-63%20Liver/1857%20Obese%20Climp63%20Liver%20-%20Segmentation",
        "raw_pattern": "Gunes_CLIMP63_8x8x8nm_3MHz.{z:04d}.tif",
        "shape": (3629, 9650, 9700),
        "seg_zips": {
            "er": "1857%20Obese%20Climp63%20ER.zip",
            "er_sheets": "1857%20Obese%20Climp63%20ER%20Sheets.zip",
            "er_tubules": "1857%20Obese%20Climp63%20ER%20Tubules.zip",
            "mito": "1857%20Obese%20Climp63%20Mitochondria.zip",
            "lipid_droplet": "1857%20Obese%20Climp63%20Lipid%20Droplet.zip",
            "nuclear_membrane": "1857%20Obese%20Climp63%20Nuclear%20membrane.zip",
        },
    },
}

PARLAKGUL_CHUNK_SHAPE = (64, 256, 256)
LabelChoice = Literal[
    "er", "er_sheets", "er_tubules", "mito", "lipid_droplet", "nuclear_membrane", "plasma_membrane"
]


def _bbox_to_str(bbox):
    return hashlib.md5("_".join(str(v) for v in bbox).encode()).hexdigest()[:12]


class _HttpFile:
    """Seekable file-like object backed by HTTP range requests."""

    def __init__(self, url):
        import requests
        self.url = url
        self._pos = 0
        r = requests.head(url, timeout=30)
        r.raise_for_status()
        self._size = int(r.headers["Content-Length"])

    def read(self, n=-1):
        import time
        import requests
        end = (self._size - 1) if n == -1 else min(self._pos + n - 1, self._size - 1)
        if self._pos > end:
            return b""
        for attempt in range(5):
            try:
                r = requests.get(self.url, headers={"Range": f"bytes={self._pos}-{end}"}, timeout=120)
                data = r.content
                self._pos += len(data)
                return data
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    def seek(self, pos, whence=0):
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = self._size + pos
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def tell(self):
        return self._pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _read_zip_slice(zip_url, slice_idx, x_min, x_max, y_min, y_max):
    """Extract one segmentation TIFF from a remote ZIP using HTTP range requests."""
    import tifffile

    zf = zipfile.ZipFile(_HttpFile(zip_url))
    names = sorted(n for n in zf.namelist() if n.endswith(".tiff") or n.endswith(".tif"))
    if slice_idx >= len(names):
        raise IndexError(f"Slice {slice_idx} out of range (zip has {len(names)} TIFFs)")
    data = zf.read(names[slice_idx])
    img = tifffile.imread(io.BytesIO(data))
    return img[y_min:y_max, x_min:x_max]


def _read_raw_slice(raw_url, x_min, x_max, y_min, y_max):
    """Download one raw TIFF slice and crop to the requested region."""
    import time
    import requests
    import tifffile

    for attempt in range(5):
        try:
            r = requests.get(raw_url, timeout=180)
            r.raise_for_status()
            img = tifffile.imread(io.BytesIO(r.content))
            return img[y_min:y_max, x_min:x_max]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def get_parlakgul_liver_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[int, int, int, int, int, int],
    sample: Literal["6461", "6464", "9430", "1857"] = "6461",
    label_choice: LabelChoice = "mito",
    download: bool = False,
) -> str:
    """Stream a subvolume from the Parlakgul liver dataset and cache it as a zarr v3 store.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        bounding_box: The region to fetch as (x_min, x_max, y_min, y_max, z_min, z_max)
            in voxel coordinates at 8 nm isotropic resolution.
        sample: Which liver sample to use. One of "6461" (lean), "6464" (obese 1),
            "9430" (obese 2), "1857" (obese Climp63).
        label_choice: Which organelle segmentation to use as labels.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        The filepath to the cached zarr store.
    """
    import zarr
    from zarr.codecs import BloscCodec

    os.makedirs(str(path), exist_ok=True)
    zarr_path = os.path.join(str(path), f"{sample}_{label_choice}_{_bbox_to_str(bounding_box)}.zarr")

    root = zarr.open_group(zarr_path, mode="a")
    if "raw" in root and "labels" in root:
        return zarr_path

    if not download:
        raise RuntimeError(
            f"No cached data found at '{zarr_path}'. Set download=True to stream it from EMPIAR."
        )

    x_min, x_max, y_min, y_max, z_min, z_max = bounding_box
    sample_info = PARLAKGUL_SAMPLES[sample]

    if label_choice not in sample_info["seg_zips"]:
        raise ValueError(f"label_choice='{label_choice}' not available for sample='{sample}'")

    shape = (z_max - z_min, y_max - y_min, x_max - x_min)
    raw_arr = np.zeros(shape, dtype=np.uint8)
    lbl_arr = np.zeros(shape, dtype=np.uint8)

    raw_base = f"{EMPIAR_BASE}/{PARLAKGUL_PAPER_BASE}/{sample_info['raw_dir']}"
    zip_name = sample_info["seg_zips"][label_choice]
    seg_zip_url = f"{EMPIAR_BASE}/{PARLAKGUL_PAPER_BASE}/{sample_info['seg_dir']}/{zip_name}"

    print(f"Streaming Parlakgul {sample} ({sample_info['condition']}) EM + {label_choice} ...")
    for i, z in enumerate(range(z_min, z_max)):
        fname = sample_info["raw_pattern"].format(z=z)
        raw_url = f"{raw_base}/{fname}"
        raw_arr[i] = _read_raw_slice(raw_url, x_min, x_max, y_min, y_max)
        lbl_arr[i] = _read_zip_slice(seg_zip_url, z, x_min, x_max, y_min, y_max)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{z_max - z_min} slices done")

    def _make_array(name, data, is_label):
        shuffle = "bitshuffle" if is_label else "shuffle"
        arr = root.create_array(
            name, shape=data.shape, chunks=PARLAKGUL_CHUNK_SHAPE, dtype=data.dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        arr[:] = data

    root.attrs["bounding_box"] = list(bounding_box)
    root.attrs["sample"] = sample
    root.attrs["label_choice"] = label_choice
    root.attrs["resolution_nm"] = [8, 8, 8]

    if "raw" not in root:
        _make_array("raw", raw_arr, is_label=False)
    if "labels" not in root:
        _make_array("labels", lbl_arr, is_label=True)

    print(f"Cached to {zarr_path} (shape {shape})")
    return zarr_path


def get_parlakgul_liver_paths(
    path: Union[os.PathLike, str],
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    sample: Literal["6461", "6464", "9430", "1857"] = "6461",
    label_choice: LabelChoice = "mito",
    download: bool = False,
) -> List[str]:
    """Get paths to Parlakgul liver zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        bounding_boxes: List of regions to fetch, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in voxel coordinates.
        sample: Which liver sample to use.
        label_choice: Which organelle segmentation to use as labels.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    return [get_parlakgul_liver_data(path, bbox, sample, label_choice, download) for bbox in bounding_boxes]


def get_parlakgul_liver_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    sample: Literal["6461", "6464", "9430", "1857"] = "6461",
    label_choice: LabelChoice = "mito",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Parlakgul liver dataset for organelle segmentation.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel coordinates.
        sample: Which liver sample to use. One of "6461", "6464", "9430", "1857".
        label_choice: Which organelle to segment.
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_parlakgul_liver_paths(path, bounding_boxes, sample, label_choice, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs,
    )


def get_parlakgul_liver_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    bounding_boxes: List[Tuple[int, int, int, int, int, int]],
    sample: Literal["6461", "6464", "9430", "1857"] = "6461",
    label_choice: LabelChoice = "mito",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for organelle segmentation in the Parlakgul liver dataset.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in 8 nm voxel coordinates.
        sample: Which liver sample to use. One of "6461", "6464", "9430", "1857".
        label_choice: Which organelle to segment.
        download: Whether to stream and cache data if not already present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_parlakgul_liver_dataset(
        path, patch_shape, bounding_boxes, sample=sample, label_choice=label_choice,
        download=download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
