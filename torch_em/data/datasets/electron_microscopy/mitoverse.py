"""MitoVerse is a collection of 3D electron microscopy volumes with mitochondria instance labels.

It brings 212 volumes from 13 source datasets (FIB-SEM, ssSEM, ssTEM and SBF-SEM, covering many species, tissues
and resolutions) into one format: each volume is a zarr store `data/<dataset>/<volume>.zarr` with the arrays
`img` (uint8, ZYX) and `mito` (instance labels, 0 is background), and the per-volume metadata (voxel size in nm
as XYZ, modality, species, tissue, provenance) is listed in `catalog.json`. The volumes of the 'han24' dataset
also have a `mask` array, which marks the annotated mossy fiber boutons: mitochondria outside of it are not labeled.

Benchmarks are defined as split files that assign whole volumes to 'train', 'val' and 'test',
see `BENCHMARKS`. The 'mitoem2.0' benchmark is the MitoEM 2.0 benchmark (see also `mitoemv2.py`) as an overlay
over the MitoVerse volumes.

The data is hosted at https://huggingface.co/datasets/pytc/MitoVerse.
The code to build it is at https://github.com/PytorchConnectomics/mitoverse.
The license differs per source dataset (see the 'source' of each volume in `catalog.json`), and the MitoVerse
authors mark several of them as to be verified, so please check them before reusing or redistributing the data.
Please cite the publications of the source datasets that you use.
"""

import os
import json
import time
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


REPO_ID = "pytc/MitoVerse"
# The data repository is updated in place, so the downloads are pinned to one revision.
REVISION = "d0af9a6fd11faaf16ce0822233b7bbf72553ce7d"
BENCHMARKS = ["mitoem2.0", "cellmap", "guay21"]
COMPLETE_MARKER = ".download_complete"
# The volumes consist of many small chunk files, so downloading many of them hits the Hugging Face rate limit.
# The download then retries after an increasing wait time. Setting the HF_TOKEN environment variable raises the limit.
N_RETRIES = 5
RETRY_WAIT = 60


def _is_transient(error):
    from huggingface_hub.errors import HfHubHTTPError

    if isinstance(error, (ConnectionError, TimeoutError)) or "429" in str(error):
        return True
    status = getattr(getattr(error, "response", None), "status_code", None)
    return isinstance(error, HfHubHTTPError) and status is not None and status >= 500


def _download(path, patterns):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("Downloading MitoVerse requires the 'huggingface_hub' package: pip install huggingface_hub")

    for attempt in range(N_RETRIES):
        try:
            snapshot_download(
                repo_id=REPO_ID, repo_type="dataset", revision=REVISION, local_dir=path, allow_patterns=patterns,
                max_workers=4,
            )
            return
        except Exception as error:
            if attempt == N_RETRIES - 1 or not _is_transient(error):
                raise
            wait = RETRY_WAIT * 2 ** attempt
            print(f"Downloading from {REPO_ID} failed ({type(error).__name__}), retrying in {wait} s.")
            time.sleep(wait)


def _download_file(path, file_name, download):
    file_path = os.path.join(path, file_name)
    if not os.path.exists(file_path):
        if not download:
            raise RuntimeError(f"Cannot find '{file_path}', but 'download' is set to False.")
        _download(path, [file_name])
    return file_path


def get_mitoverse_catalog(path: Union[os.PathLike, str], download: bool = False) -> Dict[str, dict]:
    """Get the MitoVerse catalog with the metadata of all volumes.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the catalog if it is not present.

    Returns:
        The catalog, mapping each volume id to its metadata (dataset_id, zarr, shape_zyx, voxel_nm, ...).
    """
    with open(_download_file(path, "catalog.json", download)) as f:
        return json.load(f)


def get_mitoverse_benchmark(
    path: Union[os.PathLike, str], benchmark: str, download: bool = False
) -> Dict[str, Union[str, list]]:
    """Get a MitoVerse benchmark split file.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        benchmark: The benchmark name, one of `BENCHMARKS`.
        download: Whether to download the split file if it is not present.

    Returns:
        The benchmark, with the list of its 'volumes' (id, zarr and split of each).
    """
    if benchmark not in BENCHMARKS:
        raise ValueError(f"'{benchmark}' is not a valid benchmark. Choose from {BENCHMARKS}.")
    with open(_download_file(path, f"splits/{benchmark}.json", download)) as f:
        return json.load(f)


def _select_volume_ids(path, catalog, datasets, volume_ids, benchmark, split, download):
    if volume_ids is not None:
        volume_ids = [volume_ids] if isinstance(volume_ids, str) else list(volume_ids)
        invalid = [volume_id for volume_id in volume_ids if volume_id not in catalog]
        if invalid:
            raise ValueError(f"The volume ids {invalid} are not in the MitoVerse catalog.")
    else:
        volume_ids = list(catalog)

    if datasets is not None:
        datasets = [datasets] if isinstance(datasets, str) else list(datasets)
        invalid = sorted(set(datasets) - {meta["dataset_id"] for meta in catalog.values()})
        if invalid:
            raise ValueError(f"The datasets {invalid} are not in the MitoVerse catalog.")
        volume_ids = [volume_id for volume_id in volume_ids if catalog[volume_id]["dataset_id"] in datasets]

    if benchmark is not None:
        volumes = get_mitoverse_benchmark(path, benchmark, download)["volumes"]
        in_benchmark = {volume["id"] for volume in volumes if split is None or volume["split"] == split}
        volume_ids = [volume_id for volume_id in volume_ids if volume_id in in_benchmark]
    elif split is not None:
        raise ValueError("A split can only be chosen together with a benchmark.")

    if not volume_ids:
        raise ValueError("No MitoVerse volume matches the selection.")
    return sorted(volume_ids)


def get_mitoverse_data(
    path: Union[os.PathLike, str], volume_ids: Sequence[str], download: bool = False
) -> List[str]:
    """Download MitoVerse volumes.

    Only the requested volumes are downloaded. A volume counts as present once its download has completed,
    so an interrupted download is resumed instead of silently leaving missing chunks.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        volume_ids: The ids of the volumes to download, see `get_mitoverse_catalog`.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the zarr stores of the volumes.
    """
    catalog = get_mitoverse_catalog(path, download)
    zarr_paths = [os.path.join(path, catalog[volume_id]["zarr"]) for volume_id in volume_ids]

    missing = [p for p in zarr_paths if not os.path.exists(os.path.join(p, COMPLETE_MARKER))]
    if missing and not download:
        raise RuntimeError(f"Cannot find {len(missing)} of the MitoVerse volumes, but 'download' is set to False.")

    # The volumes are downloaded per source dataset, so that the progress is kept if a later download fails.
    by_dataset = {}
    for zarr_path in missing:
        by_dataset.setdefault(os.path.dirname(zarr_path), []).append(zarr_path)
    for dataset_dir, dataset_paths in by_dataset.items():
        print(f"Downloading {len(dataset_paths)} MitoVerse volumes of '{os.path.basename(dataset_dir)}'.")
        _download(path, [f"{os.path.relpath(p, path)}/*" for p in dataset_paths])
        for zarr_path in dataset_paths:
            open(os.path.join(zarr_path, COMPLETE_MARKER), "w").close()

    return zarr_paths


def get_mitoverse_paths(
    path: Union[os.PathLike, str],
    datasets: Optional[Union[str, Sequence[str]]] = None,
    volume_ids: Optional[Union[str, Sequence[str]]] = None,
    benchmark: Optional[str] = None,
    split: Optional[Literal["train", "val", "test"]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the MitoVerse data.

    The volumes can be selected by source dataset, by volume id and by benchmark split; the selections combine.
    Without any selection all volumes are used, which downloads the whole collection (about 28 GB).

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        datasets: The source datasets to use, e.g. 'casser20' or 'han24'. By default all datasets are used.
        volume_ids: The volumes to use, see `get_mitoverse_catalog`. By default all volumes are used.
        benchmark: A benchmark that restricts the volumes to its own, one of `BENCHMARKS`.
        split: The split of the benchmark to use, one of 'train', 'val' or 'test'. Requires a benchmark.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the zarr stores of the selected volumes.
    """
    catalog = get_mitoverse_catalog(path, download)
    volume_ids = _select_volume_ids(path, catalog, datasets, volume_ids, benchmark, split, download)
    return get_mitoverse_data(path, volume_ids, download)


def get_mitoverse_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    datasets: Optional[Union[str, Sequence[str]]] = None,
    volume_ids: Optional[Union[str, Sequence[str]]] = None,
    benchmark: Optional[str] = None,
    split: Optional[Literal["train", "val", "test"]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs
) -> Dataset:
    """Get the MitoVerse dataset for mitochondria segmentation in EM.

    The volumes have different voxel sizes, see 'voxel_nm' in `get_mitoverse_catalog`.
    Without any selection all volumes are used, which downloads the whole collection (about 28 GB).

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        datasets: The source datasets to use, e.g. 'casser20' or 'han24'. By default all datasets are used.
        volume_ids: The volumes to use, see `get_mitoverse_catalog`. By default all volumes are used.
        benchmark: A benchmark that restricts the volumes to its own, one of `BENCHMARKS`.
        split: The split of the benchmark to use, one of 'train', 'val' or 'test'. Requires a benchmark.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_mitoverse_paths(path, datasets, volume_ids, benchmark, split, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=paths, raw_key="img", label_paths=paths, label_key="mito", patch_shape=patch_shape, **kwargs
    )


def get_mitoverse_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    datasets: Optional[Union[str, Sequence[str]]] = None,
    volume_ids: Optional[Union[str, Sequence[str]]] = None,
    benchmark: Optional[str] = None,
    split: Optional[Literal["train", "val", "test"]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MitoVerse dataloader for mitochondria segmentation in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        datasets: The source datasets to use, e.g. 'casser20' or 'han24'. By default all datasets are used.
        volume_ids: The volumes to use, see `get_mitoverse_catalog`. By default all volumes are used.
        benchmark: A benchmark that restricts the volumes to its own, one of `BENCHMARKS`.
        split: The split of the benchmark to use, one of 'train', 'val' or 'test'. Requires a benchmark.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mitoverse_dataset(
        path=path, patch_shape=patch_shape, datasets=datasets, volume_ids=volume_ids, benchmark=benchmark, split=split,
        download=download, offsets=offsets, boundaries=boundaries, binary=binary, **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
