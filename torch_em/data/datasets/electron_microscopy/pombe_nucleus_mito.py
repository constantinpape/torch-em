"""This dataset provides nucleus and mitochondrion segmentation masks for one cryo-ET tomogram
of a Schizosaccharomyces pombe cell.

The data is hosted on the CryoET Data Portal at https://cryoetdataportal.czscience.com/datasets/10001
(EMPIAR-10988), part of the DeePiCt benchmark family - the same paper family as the actin-only
`deepict.py` loader, but a different dataset within it (`deepict.py` hardcodes dataset 10002, the
actin dataset; this module covers dataset 10001's run "TS_0006"). The full dataset has 10 runs and
several more annotated structures (cytoplasm, vesicle, endoplasmic reticulum, nuclear envelope,
Golgi apparatus, membrane); only run "TS_0006"'s nucleus and mitochondrion masks are provided here,
since that is what was verified and visually reviewed before this loader was added.

Both masks are real, expert-verified ground truth (`groundTruthStatus=true`, `methodType=hybrid`
per the portal's own metadata), not automated predictions. Data is CC0-1.0, per the CryoET Data
Portal's portal-wide terms of use. No public corresponding-author email could be found for this
dataset (the corresponding authors, Julia Mahamid and Judith B. Zaugg, have no email listed on the
portal or reachable through a public search).
"""

import os
from typing import Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


BUCKET = "cryoet-data-portal-public"
RUN_PREFIX = f"s3://{BUCKET}/10001/TS_0006/Reconstructions/VoxelSpacing13.480"
RAW_URL = f"{RUN_PREFIX}/Tomograms/100/TS_0006.zarr"
NUCLEUS_URL = f"{RUN_PREFIX}/Annotations/108/nucleus-1.0_segmentationmask.zarr"
MITOCHONDRION_URL = f"{RUN_PREFIX}/Annotations/103/mitochondrion-1.0_segmentationmask.zarr"


def _open_remote_zarr(s3_path):
    import zarr
    from zarr.storage import FsspecStore

    store = FsspecStore.from_url(s3_path, storage_options={"anon": True}, read_only=True)
    return zarr.open(store, mode="r")


def get_pombe_nucleus_mito_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the S. pombe cryo-ET tomogram and its nucleus and mitochondrion masks.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the cached zarr store.
    """
    import zarr
    from zarr.codecs import BloscCodec

    os.makedirs(str(path), exist_ok=True)
    zarr_path = os.path.join(str(path), "TS_0006.zarr")

    root = zarr.open_group(zarr_path, mode="a")
    if "raw" in root and "nucleus" in root and "mitochondrion" in root:
        return zarr_path

    if not download:
        raise RuntimeError(f"No cached data found at '{zarr_path}'. Set download=True to stream it from S3.")

    print("Streaming the S. pombe TS_0006 tomogram and masks from the CryoET Data Portal ...")
    raw = _open_remote_zarr(f"{RAW_URL}/0")[:]
    nucleus = _open_remote_zarr(f"{NUCLEUS_URL}/0")[:]
    mitochondrion = _open_remote_zarr(f"{MITOCHONDRION_URL}/0")[:]

    assert raw.shape == nucleus.shape == mitochondrion.shape, (
        f"Shape mismatch: raw {raw.shape}, nucleus {nucleus.shape}, mitochondrion {mitochondrion.shape}"
    )

    def _make_array(name, data, shuffle):
        arr = root.create_array(
            name, shape=data.shape, chunks=tuple(min(128, s) for s in data.shape), dtype=data.dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        arr[:] = data

    root.attrs["dataset_id"] = 10001
    root.attrs["run"] = "TS_0006"
    root.attrs["ground_truth"] = True
    root.attrs["raw_source"] = RAW_URL
    root.attrs["nucleus_source"] = NUCLEUS_URL
    root.attrs["mitochondrion_source"] = MITOCHONDRION_URL

    _make_array("raw", raw, shuffle="shuffle")
    _make_array("nucleus", nucleus, shuffle="bitshuffle")
    _make_array("mitochondrion", mitochondrion, shuffle="bitshuffle")

    print(f"Cached the S. pombe TS_0006 data to '{zarr_path}' (shape {raw.shape}).")
    return zarr_path


def get_pombe_nucleus_mito_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    target: str = "nucleus",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the S. pombe dataset for nucleus or mitochondrion segmentation.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        target: The segmentation target, either 'nucleus' or 'mitochondrion'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3
    if target not in ("nucleus", "mitochondrion"):
        raise ValueError(f"'target' must be 'nucleus' or 'mitochondrion', got '{target}'.")

    zarr_path = get_pombe_nucleus_mito_data(path, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=zarr_path,
        raw_key="raw",
        label_paths=zarr_path,
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
    """Get the DataLoader for nucleus or mitochondrion segmentation in the S. pombe dataset.

    Args:
        path: Filepath to a folder where the cached zarr store will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        target: The segmentation target, either 'nucleus' or 'mitochondrion'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pombe_nucleus_mito_dataset(path, patch_shape, target=target, download=download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
