"""The human liver EM dataset contains multiscale SBF-SEM images of human liver tissue
with semantic segmentations of cellular and organelle structures.

The dataset covers a reconstructed human periportal liver volume (597 z-slices,
20000 x 20000 pixels per slice) with 9 binary semantic segmentation classes
(0=background, 255=foreground - not instance segmentation).

Currently recommended label choices for training: "er", "mito", "nucleus".
These are the classes that are well-represented and have clear biological meaning
at the imaging resolution of this dataset.

All 9 available classes:
- "bile_duct": bile duct (sparse, often absent in a given crop)
- "cell_boundary": cell boundary region (coarse, marks interior near cell edge)
- "cholangiocyte": cholangiocyte cells (sparse)
- "endothelial": endothelial cells (sparse)
- "er": endoplasmic reticulum (recommended)
- "hepatocyte": hepatocyte cell interior - tissue-level mask (large filled regions)
- "mito": mitochondria (recommended)
- "nucleus": nucleus (recommended)
- "sinusoid": sinusoidal capillary (sparse)

NOTE (on other organelles): the 9 classes above are tissue/cell-level annotations.
Organelle-level segmentations for additional structures (lipid droplets, Golgi, etc.)
are not available in EMPIAR-13356. The Parlakgul liver dataset (EMPIAR-10791) provides
richer organelle annotations (ER sheets/tubules, lipid droplets, nuclear membrane,
plasma membrane) at higher FIB-SEM resolution for mouse liver.

NOTE (on resolution): the pixel size is not documented in EMPIAR-13356. Based on the
visible tissue scale (~1-2mm tissue spanning 20000 pixels), xy resolution is estimated
at ~50-100nm/pixel. The z section thickness is also unconfirmed. Check the paper for
the exact values before selecting patch shapes for isotropic training.

Data is streamed lazily from EMPIAR-13356 via HTTP: raw 16-bit TIFFs and binary PNG
masks are fetched per z-slice and cached in a single zarr v3 store per bounding box.
All 9 label classes are stored together (raw, bile_duct, cell_boundary, ..., sinusoid).
The `label_choice` parameter in the loader selects which array to use as labels.

Bounding boxes are specified as (x_min, x_max, y_min, y_max, z_min, z_max) in voxels.
The full volume is (597, 20000, 20000) voxels (z, y, x). Tissue spans roughly
x=[1195, 18890], y=[469, 19570] - the volume edges are empty.

This dataset is from the publication https://www.biorxiv.org/content/10.64898/2026.04.22.719970v1.
Please cite it if you use this dataset in your research.

The data is publicly available at https://www.ebi.ac.uk/empiar/EMPIAR-13356/.
"""

import hashlib
import io
import os
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch_em
from .. import util


EMPIAR_BASE = "https://ftp.ebi.ac.uk/empiar/world_availability/13356/data"

HUMAN_LIVER_EM_LABEL_DIRS = {
    "er": "humanliver_er_mask",
    "mito": "humanliver_mito_mask",
    "nucleus": "humanliver_nucleus_mask",
}

HUMAN_LIVER_EM_SHAPE = (597, 20000, 20000)
# Tissue spans x=[1195,18890], y=[469,19570] - edges are empty background.
HUMAN_LIVER_EM_TISSUE_BBOX = (1195, 18890, 469, 19570, 0, 597)

# Zarr layout for bbox crops.
HUMAN_LIVER_EM_CHUNK_SHAPE = (64, 256, 256)
# Zarr layout for full-volume sharded store.
# Shards: (64, 4096, 4096) outer; chunks: (8, 256, 256) inner.
HUMAN_LIVER_EM_SHARD_SHAPE = (64, 4096, 4096)
HUMAN_LIVER_EM_INNER_CHUNK = (8, 256, 256)

LabelChoice = Literal["er", "mito", "nucleus"]


def _bbox_to_str(bbox):
    return hashlib.md5("_".join(str(v) for v in bbox).encode()).hexdigest()[:12]


class _HttpFile:
    """Seekable file-like object backed by HTTP range requests for efficient partial TIFF reading."""

    def __init__(self, url):
        import requests
        self.url = url
        self._pos = 0
        r = requests.head(url, timeout=30)
        r.raise_for_status()
        self._size = int(r.headers["Content-Length"])

    def read(self, n=-1):
        import requests
        end = (self._size - 1) if n == -1 else min(self._pos + n - 1, self._size - 1)
        if self._pos > end:
            return b""
        r = requests.get(self.url, headers={"Range": f"bytes={self._pos}-{end}"}, timeout=120)
        data = r.content
        self._pos += len(data)
        return data

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


def _read_raw_slice(z, x_min, x_max, y_min, y_max):
    """Read a cropped region from a remote TIFF using a single HTTP range request covering
    only the strips needed for the y range, avoiding downloading the full ~880 MB file."""
    import requests
    import tifffile

    url = f"{EMPIAR_BASE}/humanliver_raw_images/humanliver_raw_{z:03d}.tif"

    # Read TIFF metadata to get strip offsets and byte counts.
    with tifffile.TiffFile(_HttpFile(url)) as tif:
        page = tif.pages[0]
        offsets = page.dataoffsets
        bytecounts = page.databytecounts
        width = page.imagewidth
        dtype = np.dtype(page.dtype)

    # One strip per row - request only strips y_min..y_max in one range request.
    from imagecodecs import lzw_decode
    start_byte = offsets[y_min]
    end_byte = offsets[y_max - 1] + bytecounts[y_max - 1] - 1
    r = requests.get(url, headers={"Range": f"bytes={start_byte}-{end_byte}"}, timeout=120)
    r.raise_for_status()
    raw_bytes = r.content

    # Predictor=2 (horizontal differencing) is used - need cumsum after LZW decode.
    from imagecodecs import delta_decode
    rows = []
    for i in range(y_min, y_max):
        strip_start = offsets[i] - start_byte
        strip_data = raw_bytes[strip_start:strip_start + bytecounts[i]]
        decoded = np.frombuffer(lzw_decode(strip_data), dtype=dtype)
        row = delta_decode(decoded, axis=-1, dist=1, out=decoded).reshape(width)
        rows.append(row[x_min:x_max])

    return np.stack(rows, axis=0)


def _read_mask_slice(label_key, z, x_min, x_max, y_min, y_max):
    import PIL.Image
    import requests
    PIL.Image.MAX_IMAGE_PIXELS = None
    label_dir = HUMAN_LIVER_EM_LABEL_DIRS[label_key]
    url = f"{EMPIAR_BASE}/{label_dir}/{label_dir}_{z:03d}.png"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    img = np.array(PIL.Image.open(io.BytesIO(r.content)))
    if img.ndim == 3:
        img = img[..., 0]
    return img[y_min:y_max, x_min:x_max]


def get_human_liver_em_data(
    path: Union[os.PathLike, str],
    bounding_box: Tuple[int, int, int, int, int, int],
    download: bool = False,
) -> str:
    """Stream a subvolume from the human liver EM dataset and cache it as a zarr v3 store.

    All 9 label classes are stored in a single zarr alongside the raw EM, so each
    bounding box is only downloaded once regardless of which label_choice is used.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        bounding_box: The region to fetch as (x_min, x_max, y_min, y_max, z_min, z_max)
            in voxel coordinates. Tissue spans roughly x=[1195,18890], y=[469,19570].
        download: Whether to stream and cache the data if it is not present.

    Returns:
        The filepath to the cached zarr store.
    """
    import zarr
    from zarr.codecs import BloscCodec

    os.makedirs(str(path), exist_ok=True)
    zarr_path = os.path.join(str(path), f"{_bbox_to_str(bounding_box)}.zarr")

    root = zarr.open_group(zarr_path, mode="a")
    all_keys = ["raw"] + list(HUMAN_LIVER_EM_LABEL_DIRS.keys())
    if all(k in root for k in all_keys):
        return zarr_path

    if not download:
        raise RuntimeError(
            f"No cached data found at '{zarr_path}'. Set download=True to stream it from EMPIAR."
        )

    x_min, x_max, y_min, y_max, z_min, z_max = bounding_box
    shape = (z_max - z_min, y_max - y_min, x_max - x_min)

    print(f"Streaming Human Liver EM + all labels for bbox {bounding_box} ...")
    raw_vol = np.zeros(shape, dtype=np.uint16)
    label_vols = {k: np.zeros(shape, dtype=np.uint8) for k in HUMAN_LIVER_EM_LABEL_DIRS}

    for i, z in enumerate(range(z_min, z_max)):
        raw_vol[i] = _read_raw_slice(z, x_min, x_max, y_min, y_max)
        for label_key in HUMAN_LIVER_EM_LABEL_DIRS:
            label_vols[label_key][i] = _read_mask_slice(label_key, z, x_min, x_max, y_min, y_max)
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{z_max - z_min} slices done")

    def _make_array(name, data, is_label):
        shuffle = "bitshuffle" if is_label else "shuffle"
        arr = root.create_array(
            name, shape=data.shape, chunks=HUMAN_LIVER_EM_CHUNK_SHAPE, dtype=data.dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        arr[:] = data

    root.attrs["bounding_box"] = list(bounding_box)

    if "raw" not in root:
        _make_array("raw", raw_vol, is_label=False)
    for label_key, vol in label_vols.items():
        if label_key not in root:
            _make_array(label_key, vol, is_label=True)

    print(f"Cached to {zarr_path} (shape {shape})")
    return zarr_path


def get_human_liver_em_full_volume(
    path: Union[os.PathLike, str],
    download: bool = False,
) -> str:
    """Download the full human liver EM tissue volume into a sharded zarr v3 store.

    Downloads all 597 z-slices for the tissue region x=[1195,18890], y=[469,19570]
    with raw EM + er/mito/nucleus labels. Data is written slice by slice to avoid
    memory issues. Estimated storage: ~100-150 GB compressed. Estimated download
    time: ~12 hours (one-time cost).

    The sharded zarr uses shard shape (64, 4096, 4096) with inner chunks (8, 256, 256),
    enabling efficient random crop access during training without loading the full volume.

    Args:
        path: Filepath to a folder where the zarr store will be saved.
        download: Whether to stream and cache the data if it is not present.

    Returns:
        The filepath to the full-volume zarr store.
    """
    import zarr
    from zarr.codecs import BloscCodec, ShardingCodec

    os.makedirs(str(path), exist_ok=True)
    zarr_path = os.path.join(str(path), "full_volume.zarr")

    x_min, x_max, y_min, y_max, z_min, z_max = HUMAN_LIVER_EM_TISSUE_BBOX
    shape = (z_max - z_min, y_max - y_min, x_max - x_min)

    root = zarr.open_group(zarr_path, mode="a")
    all_keys = ["raw"] + list(HUMAN_LIVER_EM_LABEL_DIRS.keys())
    if all(k in root for k in all_keys):
        return zarr_path

    if not download:
        raise RuntimeError(
            f"Full-volume zarr not found at '{zarr_path}'. Set download=True to stream from EMPIAR."
            " Note: download takes ~12 hours and requires ~100-150 GB disk space."
        )

    def _make_sharded(name, dtype, is_label):
        shuffle = "bitshuffle" if is_label else "shuffle"
        return root.create_array(
            name, shape=shape, chunks=HUMAN_LIVER_EM_SHARD_SHAPE, dtype=dtype,
            compressors=ShardingCodec(
                chunk_shape=HUMAN_LIVER_EM_INNER_CHUNK,
                codecs=[BloscCodec(cname="zstd", clevel=6, shuffle=shuffle)],
            ),
        )

    if "raw" not in root:
        _make_sharded("raw", np.dtype("uint16"), is_label=False)
    for label_key in HUMAN_LIVER_EM_LABEL_DIRS:
        if label_key not in root:
            _make_sharded(label_key, np.dtype("uint8"), is_label=True)

    root.attrs["tissue_bbox"] = list(HUMAN_LIVER_EM_TISSUE_BBOX)
    n_slices = z_max - z_min
    print(f"Streaming full Human Liver EM volume ({shape}) - this will take several hours ...")

    for i, z in enumerate(range(z_min, z_max)):
        raw_slice = _read_raw_slice(z, x_min, x_max, y_min, y_max)
        root["raw"][i] = raw_slice
        for label_key in HUMAN_LIVER_EM_LABEL_DIRS:
            mask_slice = _read_mask_slice(label_key, z, x_min, x_max, y_min, y_max)
            root[label_key][i] = mask_slice
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_slices} slices done")

    print(f"Full volume cached to {zarr_path} (shape {shape})")
    return zarr_path


def get_human_liver_em_paths(
    path: Union[os.PathLike, str],
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    download: bool = False,
    full_volume: bool = False,
) -> List[str]:
    """Get paths to human liver EM zarr stores.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        bounding_boxes: List of regions to fetch, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in voxel coordinates.
            Ignored when full_volume=True.
        download: Whether to stream and cache the data if it is not present.
        full_volume: If True, download/use the full tissue volume as a single sharded
            zarr v3 store (~12h download, ~100-150 GB). Supersedes bounding_boxes.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    if full_volume:
        return [get_human_liver_em_full_volume(path, download)]
    if bounding_boxes is None:
        raise ValueError("Provide bounding_boxes or set full_volume=True.")
    return [get_human_liver_em_data(path, bbox, download) for bbox in bounding_boxes]


def get_human_liver_em_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    label_choice: LabelChoice = "mito",
    download: bool = False,
    full_volume: bool = False,
    **kwargs,
) -> Dataset:
    """Get the human liver EM dataset for semantic segmentation.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training. The pixel
            resolution is unconfirmed (estimated ~50-100 nm/px xy). Check the
            paper for the exact values when selecting isotropic patch shapes.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in voxel coordinates.
            Ignored when full_volume=True.
        label_choice: Which structure to segment.
        download: Whether to stream and cache data if not already present.
        full_volume: If True, use the full sharded tissue volume (~12h one-time download).
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_human_liver_em_paths(path, bounding_boxes, download, full_volume)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key=label_choice,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_human_liver_em_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    bounding_boxes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    label_choice: LabelChoice = "mito",
    download: bool = False,
    full_volume: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for semantic segmentation in the human liver EM dataset.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        bounding_boxes: List of subvolumes to use, each as
            (x_min, x_max, y_min, y_max, z_min, z_max) in voxel coordinates.
            Ignored when full_volume=True.
        label_choice: Which structure to segment.
        download: Whether to stream and cache data if not already present.
        full_volume: If True, use the full sharded tissue volume (~12h one-time download).
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_human_liver_em_dataset(
        path, patch_shape, bounding_boxes, label_choice=label_choice,
        download=download, full_volume=full_volume, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
