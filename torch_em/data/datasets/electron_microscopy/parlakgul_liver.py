"""The Parlakgul liver dataset contains FIB-SEM volumes of mouse liver with dense
semantic segmentation of 7 organelle classes. All labels are binary semantic masks
(0=background, 255=foreground) - not instance segmentation.

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
- "nuclear_membrane": nuclear membrane (not available for 1857)
- "plasma_membrane": plasma membrane

Data is streamed lazily from EMPIAR-10791 via HTTP range requests: only the rows of the raw TIFFs and the
segmentation slices (from the ZIP archives) of the requested bounding box are fetched and cached as zarr v3.

Bounding boxes are specified as (x_min, x_max, y_min, y_max, z_min, z_max) in voxels.

This dataset is from the publication https://doi.org/10.1038/s41586-022-04518-2.
Please cite it if you use this dataset in your research.

The data is publicly available at https://www.ebi.ac.uk/empiar/EMPIAR-10791/.
"""

import hashlib
import io
import os
import re
import shutil
import zipfile
from concurrent import futures
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

# The raw file of slice z is 'raw_pattern' formatted with z + 'raw_z_offset' (the raw files of 6461 are numbered
# from 0, the others from 1). The segmentation slices are found by the slice number in their name, see
# '_index_zip_slices'.
PARLAKGUL_SAMPLES: Dict[str, dict] = {
    "6461": {
        "condition": "lean",
        "raw_dir": "6461%20-%20Lean%20Liver/6461%20Lean%20Liver%20-%20Raw",
        "seg_dir": "6461%20-%20Lean%20Liver/6461%20Lean%20Liver%20-%20Segmentation",
        "raw_pattern": "Gunes_WT1_8x8x8nm_3MHz.{z:04d}.tif",
        "raw_z_offset": 0,
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
        "raw_pattern": "6464%20Ob%20Fed%20Raw%20with%20All_T001_Z{z:03d}_C09.tif",
        "raw_z_offset": 1,
        "shape": (7896, 10200, 9112),
        "seg_zips": {
            "er": "6464%20Obese1%20ER.zip",
            "er_sheets": "6464%20Obese1%20ER%20Sheets.zip",
            "er_tubules": "6464%20Obese1%20ER%20Tubules.zip",
            "mito": "6464%20Obese1%20Mitochondria.zip",
            "lipid_droplet": "6464%20Obese1%20Lipid%20Droplet.zip",
            "nuclear_membrane": "6464%20Obese1%20Nuclear%20membrane.zip",
            "plasma_membrane": "6464%20Obese1%20Plasma%20membrane.zip",
        },
    },
    "9430": {
        "condition": "obese2",
        "raw_dir": "9430%20-%20Obese2%20Liver/9430%20Obese2%20Liver%20-%20Raw/9430%20raw",
        "seg_dir": "9430%20-%20Obese2%20Liver/9430%20Obese2%20Liver%20-%20Segmentation",
        "raw_pattern": "9430%20Ob%20Fed%20Raw_T001_Z{z:03d}_C07.tif",
        "raw_z_offset": 1,
        "shape": (8501, 8050, 8000),
        "seg_zips": {
            "er": "9430%20Obese2%20ER.zip",
            "er_sheets": "9430%20Obese2%20ER%20Sheets.zip",
            "er_tubules": "9430%20Obese2%20ER%20Tubules.zip",
            "mito": "9430%20Obese2%20Mitochondria.zip",
            "lipid_droplet": "9430%20Obese2%20Lipid%20Droplet.zip",
            "nuclear_membrane": "9430%20Obese2%20nuclear%20membrane.zip",
            "plasma_membrane": "9430%20Obese2%20Plasma%20membrane.zip",
        },
    },
    "1857": {
        "condition": "obese_climp63",
        "raw_dir": "1857%20-%20Obese%20Climp-63%20Liver/1857%20Obese%20Climp-63%20Raw",
        "seg_dir": "1857%20-%20Obese%20Climp-63%20Liver/1857%20Obese%20Climp-63%20Segmentation",
        "raw_pattern": "1857_Obese_Climp63_raw_T001_Z{z:03d}_C07.tif",
        "raw_z_offset": 1,
        "shape": (3629, 9650, 9700),
        "seg_zips": {
            "er": "1857_Obese_Climp63_ER.zip",
            "er_sheets": "1857_Obese_Climp63_ER-sheets.zip",
            "er_tubules": "1857_Obese_Climp63_ER-tubules.zip",
            "mito": "1857_Obese_Climp63_Mitochondria.zip",
            "lipid_droplet": "1857_Obese_Climp63_Lipid-droplet.zip",
            "plasma_membrane": "1857_Obese_Climp63_Plasma-membrane.zip",
        },
    },
}

PARLAKGUL_CHUNK_SHAPE = (64, 256, 256)
N_THREADS = 4
LabelChoice = Literal[
    "er", "er_sheets", "er_tubules", "mito", "lipid_droplet", "nuclear_membrane", "plasma_membrane"
]


def _bbox_to_str(bbox):
    return hashlib.md5("_".join(str(v) for v in bbox).encode()).hexdigest()[:12]


class _HttpFile:
    """Seekable file-like object backed by HTTP range requests.

    Reading a TIFF or ZIP header issues many small reads, so small reads are served from cached 1 MB blocks.
    Large reads are fetched with a single range request.
    """
    block_size = 1 << 20
    n_cached_blocks = 8

    def __init__(self, url):
        import requests
        self.url = url
        self._pos = 0
        self._blocks = {}
        r = requests.head(url, timeout=30)
        r.raise_for_status()
        self._size = int(r.headers["Content-Length"])

    def _fetch(self, start, stop):
        import time
        import requests
        for attempt in range(5):
            try:
                r = requests.get(self.url, headers={"Range": f"bytes={start}-{stop - 1}"}, timeout=300)
                r.raise_for_status()
                if len(r.content) != stop - start:
                    raise IOError(f"Expected {stop - start} bytes from {self.url}, got {len(r.content)}.")
                return r.content
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    def _block(self, index):
        if index not in self._blocks:
            if len(self._blocks) >= self.n_cached_blocks:
                self._blocks.pop(next(iter(self._blocks)))
            start = index * self.block_size
            self._blocks[index] = self._fetch(start, min(start + self.block_size, self._size))
        return self._blocks[index]

    def read(self, n=-1):
        stop = self._size if n is None or n < 0 else min(self._pos + n, self._size)
        if self._pos >= stop:
            return b""
        if stop - self._pos > self.block_size:
            data = self._fetch(self._pos, stop)
        else:
            first, last = self._pos // self.block_size, (stop - 1) // self.block_size
            data = b"".join(self._block(index) for index in range(first, last + 1))
            data = data[self._pos - first * self.block_size:stop - first * self.block_size]
        self._pos = stop
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


def _index_zip_slices(zf):
    """Map the z index of each segmentation slice to its member in the ZIP archive.

    The archives number their slices differently ('0000.tiff', '000' to '7895' without consistent zero padding,
    or '..._Z001_C01.tif' starting at 1), and do not always store them in z order. So the z index is parsed from
    the slice number in the member name, relative to the smallest slice number.
    """
    numbers = {}
    for info in zf.infolist():
        if not info.filename.lower().endswith((".tif", ".tiff")):
            continue
        stem = os.path.splitext(os.path.basename(info.filename))[0]
        match = re.search(r"Z(\d+)_C\d+$", stem) or re.search(r"(\d+)$", stem)
        numbers[int(match.group(1))] = info
    first = min(numbers)
    return {number - first: info for number, info in numbers.items()}


def _read_zip_slice(zf, info, x_min, x_max, y_min, y_max):
    """Extract one segmentation slice from the remote ZIP archive and crop it to the requested region."""
    import tifffile
    return tifffile.imread(io.BytesIO(zf.read(info)))[y_min:y_max, x_min:x_max]


def _read_raw_slice(raw_url, x_min, x_max, y_min, y_max):
    """Read the rows y_min:y_max of one raw TIFF slice and crop them to the requested region.

    The raw TIFFs are uncompressed with contiguous strips, so the rows are fetched with a single range request.
    Other layouts fall back to reading the whole slice.
    """
    import tifffile

    fh = _HttpFile(raw_url)
    with tifffile.TiffFile(fh) as tif:
        page = tif.pages[0]
        width, rows_per_strip = page.shape[1], page.rowsperstrip
        offsets = np.asarray(page.dataoffsets)
        contiguous = page.compression == 1 and page.dtype == np.uint8 and page.samplesperpixel == 1 and bool(
            np.all(np.diff(offsets) == rows_per_strip * width)
        )
    if not contiguous:
        fh.seek(0)
        return tifffile.imread(io.BytesIO(fh.read()))[y_min:y_max, x_min:x_max]

    fh.seek(int(offsets[y_min // rows_per_strip]) + (y_min % rows_per_strip) * width)
    rows = np.frombuffer(fh.read((y_max - y_min) * width), dtype="uint8").reshape(y_max - y_min, width)
    return rows[:, x_min:x_max]


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

    if sample not in PARLAKGUL_SAMPLES:
        raise ValueError(f"'{sample}' is not a valid sample. Choose from {list(PARLAKGUL_SAMPLES)}.")
    sample_info = PARLAKGUL_SAMPLES[sample]
    if label_choice not in sample_info["seg_zips"]:
        raise ValueError(f"label_choice='{label_choice}' not available for sample='{sample}'")

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
    raw_arr = np.zeros(shape, dtype=np.uint8)
    lbl_arr = np.zeros(shape, dtype=np.uint8)

    raw_base = f"{EMPIAR_BASE}/{PARLAKGUL_PAPER_BASE}/{sample_info['raw_dir']}"
    zip_name = sample_info["seg_zips"][label_choice]
    seg_zip_url = f"{EMPIAR_BASE}/{PARLAKGUL_PAPER_BASE}/{sample_info['seg_dir']}/{zip_name}"

    print(f"Streaming Parlakgul {sample} ({sample_info['condition']}) EM + {label_choice} ...")

    # The ZIP directory is read once, the segmentation slices are then fetched one by one from it.
    zf = zipfile.ZipFile(_HttpFile(seg_zip_url))
    slices = _index_zip_slices(zf)
    if len(slices) != n_z:
        raise RuntimeError(f"Expected {n_z} segmentation slices in {zip_name}, found {len(slices)}.")

    def _fetch_raw(i):
        url = f"{raw_base}/{sample_info['raw_pattern'].format(z=z_min + i + sample_info['raw_z_offset'])}"
        raw_arr[i] = _read_raw_slice(url, x_min, x_max, y_min, y_max)

    with futures.ThreadPoolExecutor(N_THREADS) as tp:
        raw_tasks = [tp.submit(_fetch_raw, i) for i in range(shape[0])]
        for i, z in enumerate(range(z_min, z_max)):
            lbl_arr[i] = _read_zip_slice(zf, slices[z], x_min, x_max, y_min, y_max)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{shape[0]} segmentation slices done")
        for task in raw_tasks:
            task.result()

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
            name, shape=data.shape, chunks=PARLAKGUL_CHUNK_SHAPE, dtype=data.dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        arr[:] = data
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    os.rename(tmp_path, zarr_path)

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
