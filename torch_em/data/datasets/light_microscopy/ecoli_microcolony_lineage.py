"""This dataset contains phase-contrast time-lapse images of growing *E. coli* microcolonies with
per-frame single-cell instance segmentation and full lineage tracking (Schnitzcells format), covering
eight genetic pathways (toxin production, SOS-stress response and metabolism).

The dataset is hosted on Zenodo at https://doi.org/10.5281/zenodo.268921.
The dataset is from the publication https://doi.org/10.1016/j.cels.2018.03.009.

Please cite it if you use this dataset for your research.
"""

import os
import re
from glob import glob
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import tifffile
import scipy.io as sio
from skimage.feature import match_template

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


GENES = ["cib", "crosstalk", "metA", "pheA", "recA", "rpsM", "SOSInteraction", "trpL"]

URLS = {gene: f"https://zenodo.org/records/268921/files/{gene}.zip" for gene in GENES}

CHECKSUMS = {
    "cib": "1e6408984f531347861dce04d9b4a4af1f51a6c85d2cf7f38149158632928555",
    "crosstalk": "dfd41c10be2f4271efaa79c1bea7c650b56769936186ec491565b5140e9a3d95",
    "metA": "c12142c15eb88c13f9e9456de02e2bb5e7c9aaadbca6651315b19504cbf11397",
    "pheA": "e92ea5af5e4900f7c2498efed1478d7948a1c195579ec211dd633e08c8b06cea",
    "recA": "145bd270d574b6210e36a9a17ab6be1fb7214498e40a1118018b3c97f91dfe28",
    "rpsM": "e0c3c49f9f4d47772f7a50a17f722bdf9031a68999e9b0fab77247e58fbb845b",
    "SOSInteraction": "aafc6c5af6ac5ad5659d6e1ddc4827321d7497080333f1e86ccfbd8eb76bf0fc",
    "trpL": "022b0ca2afe94d1890251bcb7c70ee500806f8c247ed2ba07ede35ed60b017fb",
}


def _load_lineage_lookup(lin_path):
    lookup = {}
    try:
        lineage = sio.loadmat(lin_path, simplify_cells=True)["schnitzcells"]
        for track_id, schnitz in enumerate(lineage, start=1):
            frames = np.atleast_1d(schnitz["frames"])
            cellnos = np.atleast_1d(schnitz["cellno"])
            for frame, cellno in zip(frames, cellnos):
                lookup[(int(frame), int(cellno))] = track_id
    except NotImplementedError:  # MATLAB v7.3 (HDF5-based) files need h5py instead of scipy.io.
        with h5py.File(lin_path, "r") as f:
            frames_ds, cellno_ds = f["schnitzcells"]["frames"], f["schnitzcells"]["cellno"]
            for track_id in range(1, frames_ds.shape[0] + 1):
                frames = np.array(f[frames_ds[track_id - 1, 0]]).ravel()
                cellnos = np.array(f[cellno_ds[track_id - 1, 0]]).ravel()
                for frame, cellno in zip(frames, cellnos):
                    lookup[(int(frame), int(cellno))] = track_id
    return lookup


def _find_crop_offset(image, template, rect, pad=20):
    y0, x0, y1, x1 = rect
    h, w = template.shape
    wy0, wy1 = max(0, y0 - 1 - pad), min(image.shape[0], y1 + pad)
    wx0, wx1 = max(0, x0 - 1 - pad), min(image.shape[1], x1 + pad)
    window = image[wy0:wy1, wx0:wx1]
    if window.shape[0] < h or window.shape[1] < w:
        return y0 - 1, x0 - 1
    result = match_template(window.astype(np.float32), template.astype(np.float32))
    dy, dx = np.unravel_index(np.argmax(result), result.shape)
    return wy0 + dy, wx0 + dx


def _reconstruct_label_frame(seg_path, raw_shape, lineage_lookup, frame_num, raw_image):
    seg = sio.loadmat(seg_path, simplify_cells=True)
    rect = seg.get("rect")
    if rect is None or len(rect) != 4:
        return None

    y0, x0, y1, x1 = [int(v) for v in rect]
    local_labels = seg["Lc"]
    phsub = seg["phsub"]

    yy0, xx0 = _find_crop_offset(raw_image, phsub, (y0, x0, y1, x1))
    h, w = local_labels.shape

    canvas = np.zeros(raw_shape, dtype=np.uint16)
    placed = np.zeros_like(local_labels, dtype=np.uint16)
    for local_id in np.unique(local_labels):
        if local_id == 0:
            continue
        global_id = lineage_lookup.get((frame_num, int(local_id)))
        if global_id is not None:
            placed[local_labels == local_id] = global_id
    canvas[yy0:yy0 + h, xx0:xx0 + w] = placed
    return canvas


def _prepare_colony_labels(colony_dir, cache_root):
    colony_name = os.path.basename(colony_dir.rstrip("/"))
    lin_path = os.path.join(colony_dir, "data", f"{colony_name}_lin.mat")
    if not os.path.exists(lin_path):
        return [], []

    label_dir = os.path.join(cache_root, colony_name)
    raw_paths_all = sorted(glob(os.path.join(colony_dir, "images", f"{colony_name}-p-*.tif")))
    seg_paths = sorted(glob(os.path.join(colony_dir, "segmentation", f"{colony_name}seg*.mat")))
    if not raw_paths_all or not seg_paths:
        return [], []

    seg_by_frame = {}
    for seg_path in seg_paths:
        fname = os.path.basename(seg_path)
        match = re.search(r"seg(\d+)\.mat$", fname)
        if match is None:
            continue
        seg_by_frame[int(match.group(1))] = seg_path

    os.makedirs(label_dir, exist_ok=True)
    lineage_lookup = _load_lineage_lookup(lin_path)

    raw_paths, label_paths = [], []
    for raw_path in raw_paths_all:
        fname = os.path.basename(raw_path)
        match = re.search(r"-p-(\d+)\.tif$", fname)
        if match is None:
            continue
        frame_num = int(match.group(1))
        seg_path = seg_by_frame.get(frame_num)
        if seg_path is None:
            continue

        label_path = os.path.join(label_dir, fname)
        if not os.path.exists(label_path):
            raw_image = tifffile.imread(raw_path)
            label = _reconstruct_label_frame(seg_path, raw_image.shape, lineage_lookup, frame_num, raw_image)
            if label is None:
                continue
            tifffile.imwrite(label_path, label)

        raw_paths.append(raw_path)
        label_paths.append(label_path)

    return raw_paths, label_paths


def get_ecoli_microcolony_lineage_data(
    path: Union[os.PathLike, str], genes: Optional[List[str]] = None, download: bool = False,
) -> List[str]:
    f"""Download the E. coli microcolony lineage dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        genes: The genetic pathways to download. The available pathways are: {', '.join(GENES)}.
            By default downloads all of them.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the folders where each pathway's data is stored.
    """
    genes = GENES if genes is None else genes
    for gene in genes:
        if gene not in GENES:
            raise ValueError(f"'{gene}' is not a valid pathway, choose one of {GENES}.")

    os.makedirs(path, exist_ok=True)

    gene_dirs = []
    for gene in genes:
        gene_dir = os.path.join(path, gene)
        if not os.path.exists(gene_dir):
            zip_path = os.path.join(path, f"{gene}.zip")
            util.download_source(path=zip_path, url=URLS[gene], download=download, checksum=CHECKSUMS[gene])
            util.unzip(zip_path=zip_path, dst=path)
        gene_dirs.append(gene_dir)

    return gene_dirs


def get_ecoli_microcolony_lineage_paths(
    path: Union[os.PathLike, str], genes: Optional[List[str]] = None, download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths for the E. coli microcolony lineage dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        genes: The genetic pathways to use. By default uses all of them.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the raw phase-contrast images.
        List of filepaths for the reconstructed instance segmentation and lineage labels.
    """
    gene_dirs = get_ecoli_microcolony_lineage_data(path, genes, download)

    raw_paths, label_paths = [], []
    for gene_dir in gene_dirs:
        cache_root = os.path.join(gene_dir, "labels_lineage")
        colony_dirs = [d for d in glob(os.path.join(gene_dir, "*")) if os.path.isdir(d) and d != cache_root]
        for colony_dir in colony_dirs:
            this_raw_paths, this_label_paths = _prepare_colony_labels(colony_dir, cache_root)
            raw_paths.extend(this_raw_paths)
            label_paths.extend(this_label_paths)

    assert raw_paths and len(raw_paths) == len(label_paths)
    return raw_paths, label_paths


def get_ecoli_microcolony_lineage_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    genes: Optional[List[str]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the E. coli microcolony lineage dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        genes: The genetic pathways to use. By default uses all of them.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_ecoli_microcolony_lineage_paths(path, genes, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        ndim=2,
        is_seg_dataset=False,
        **kwargs
    )


def get_ecoli_microcolony_lineage_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    genes: Optional[List[str]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the E. coli microcolony lineage dataloader for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        genes: The genetic pathways to use. By default uses all of them.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ecoli_microcolony_lineage_dataset(path, patch_shape, genes, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
