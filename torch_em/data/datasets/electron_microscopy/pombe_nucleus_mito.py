"""This dataset provides nucleus, mitochondrion, and lipid droplet segmentation masks for cryo-ET
tomograms of Schizosaccharomyces pombe cells.

The data is hosted on the CryoET Data Portal, part of the DeePiCt benchmark family - the same
paper family as the actin-only `deepict.py` loader, but different datasets within it (`deepict.py`
hardcodes dataset 10002, the actin dataset). This module covers:

- Dataset 10001 (https://cryoetdataportal.czscience.com/datasets/10001, EMPIAR-10988), run
  "TS_0006": nucleus and mitochondrion.
- Dataset 10000 (https://cryoetdataportal.czscience.com/datasets/10000), run "TS_045": nucleus
  and mitochondrion. Run "TS_028": lipid droplet.

Both datasets have many more runs and several more annotated structures (cytoplasm, vesicle,
endoplasmic reticulum, nuclear envelope, Golgi apparatus, membrane); only the runs and targets
listed above are provided here, since that is what was verified and visually reviewed before
this loader was added.

All masks are real, expert-verified ground truth (`groundTruthStatus=true`, `methodType=hybrid`
per the portal's own metadata), not automated predictions. Data is CC0-1.0, per the CryoET Data
Portal's portal-wide terms of use. The corresponding author for dataset 10000, Julia Mahamid, has
a public email (julia.mahamid@embl.de); no public email could be found for Judith B. Zaugg or for
dataset 10001's corresponding authors despite a real search.
"""

import os
from typing import List, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


BUCKET = "cryoet-data-portal-public"

# (dataset_id, run) -> {target: annotation folder}. A run's zarr group caches "raw" plus one
# array per target listed here, so a run contributing multiple targets (e.g. TS_045's nucleus and
# mitochondrion) is only downloaded once regardless of which target is requested.
RUN_LABELS = {
    (10001, "TS_0006"): {"nucleus": "108", "mitochondrion": "103"},
    (10000, "TS_045"): {"nucleus": "108", "mitochondrion": "103"},
    (10000, "TS_028"): {"lipid_droplet": "110"},
}
TARGETS = ("nucleus", "mitochondrion", "lipid_droplet")


def _open_remote_zarr(s3_path):
    import zarr
    from zarr.storage import FsspecStore

    store = FsspecStore.from_url(s3_path, storage_options={"anon": True}, read_only=True)
    return zarr.open(store, mode="r")


def _run_prefix(dataset_id, run):
    return f"s3://{BUCKET}/{dataset_id}/{run}/Reconstructions/VoxelSpacing13.480"


def get_pombe_nucleus_mito_data(
    path: Union[os.PathLike, str], dataset_id: int, run: str, download: bool = False
) -> str:
    """Download one S. pombe cryo-ET run's tomogram and its available organelle masks.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        dataset_id: The CryoET Data Portal dataset id, one of the keys in `RUN_LABELS`.
        run: The run name, matching `dataset_id` in `RUN_LABELS`.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the cached zarr store.
    """
    import zarr
    from zarr.codecs import BloscCodec

    key = (dataset_id, run)
    if key not in RUN_LABELS:
        raise ValueError(f"'{key}' is not a valid (dataset_id, run). Choose from {sorted(RUN_LABELS.keys())}.")
    labels = RUN_LABELS[key]

    os.makedirs(str(path), exist_ok=True)
    zarr_path = os.path.join(str(path), f"{dataset_id}_{run}.zarr")

    root = zarr.open_group(zarr_path, mode="a")
    if "raw" in root and all(name in root for name in labels):
        return zarr_path

    if not download:
        raise RuntimeError(f"No cached data found at '{zarr_path}'. Set download=True to stream it from S3.")

    print(f"Streaming the S. pombe {run} tomogram and masks from the CryoET Data Portal ...")
    prefix = _run_prefix(dataset_id, run)
    raw = _open_remote_zarr(f"{prefix}/Tomograms/100/{run}.zarr/0")[:]

    def _make_array(name, data, shuffle):
        arr = root.create_array(
            name, shape=data.shape, chunks=tuple(min(128, s) for s in data.shape), dtype=data.dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        arr[:] = data

    if "raw" not in root:
        _make_array("raw", raw, shuffle="shuffle")

    for name, folder in labels.items():
        if name in root:
            continue
        label_data = _open_remote_zarr(f"{prefix}/Annotations/{folder}/{name}-1.0_segmentationmask.zarr/0")[:]
        assert label_data.shape == raw.shape, (
            f"Shape mismatch for {key}, target '{name}': raw {raw.shape} vs label {label_data.shape}"
        )
        _make_array(name, label_data, shuffle="bitshuffle")

    root.attrs["dataset_id"] = dataset_id
    root.attrs["run"] = run
    root.attrs["ground_truth"] = True

    print(f"Cached the S. pombe {run} data to '{zarr_path}' (shape {raw.shape}).")
    return zarr_path


def get_pombe_nucleus_mito_paths(
    path: Union[os.PathLike, str], target: str = "nucleus", download: bool = False,
) -> List[str]:
    """Get paths to cached S. pombe zarr stores that provide the requested target.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        target: The segmentation target, one of 'nucleus', 'mitochondrion', or 'lipid_droplet'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the cached zarr stores.
    """
    if target not in TARGETS:
        raise ValueError(f"'target' must be one of {TARGETS}, got '{target}'.")
    runs = [key for key, labels in RUN_LABELS.items() if target in labels]
    return [get_pombe_nucleus_mito_data(path, dataset_id, run, download) for dataset_id, run in runs]


def get_pombe_nucleus_mito_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    target: str = "nucleus",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the S. pombe dataset for nucleus, mitochondrion, or lipid droplet segmentation.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        target: The segmentation target, one of 'nucleus', 'mitochondrion', or 'lipid_droplet'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_pombe_nucleus_mito_paths(path, target, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key=target,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_pombe_nucleus_mito_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    target: str = "nucleus",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for nucleus, mitochondrion, or lipid droplet segmentation in the S. pombe dataset.

    Args:
        path: Filepath to a folder where the cached zarr stores will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        target: The segmentation target, one of 'nucleus', 'mitochondrion', or 'lipid_droplet'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pombe_nucleus_mito_dataset(path, patch_shape, target=target, download=download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
