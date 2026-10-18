"""The DeePiCt S. pombe dataset contains expert-verified segmentations of cellular structures in cryo-ET.
It consists of 10 tomograms of S. pombe cryo-FIB lamellae (collected with defocus) at 13.48 Angstrom voxel spacing,
which are annotated with up to 12 different structures: cytoplasm, vesicle, mitochondrion, microtubule,
endoplasmic reticulum, nuclear envelope, nucleus, vacuole, lipid droplet, Golgi apparatus, multivesicular body
and membrane. Not all structures are annotated in every tomogram, see `RUN_CLASSES`.

The data is hosted on the CryoET Data Portal at https://cryoetdataportal.czscience.com/datasets/10000.
The dataset is part of the publication https://doi.org/10.1038/s41592-022-01746-2.
Please cite it if you use this dataset in your research.
"""

import os
from typing import List, Literal, Tuple, Union

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


DATASET_ID = 10000
VOXEL_SPACING = "VoxelSpacing13.480"
TOMOGRAM_FOLDER = "100"

# The structures with their portal annotation folder. The same folder is used for all runs.
# The order defines the label ids of the 'all' target (starting at 1); later structures overwrite earlier ones.
CLASSES = {
    "cytoplasm": "102",
    "mitochondrion": "103",
    "vesicle": "104",
    "microtubule": "105",
    "endoplasmic_reticulum": "106",
    "nuclear_envelope": "107",
    "nucleus": "108",
    "vacuole": "109",
    "lipid_droplet": "110",
    "golgi_apparatus": "111",
    "multivesicular_body": "112",
    "membrane": "114",
}

# The membrane overlaps with the other structures and is only annotated in some runs.
# It is therefore not part of the 'all' target, but only available as an individual target.
ALL_CLASSES = tuple(name for name in CLASSES if name != "membrane")

# The ground-truth structures that are annotated in each run.
RUN_CLASSES = {
    "TS_026": ("cytoplasm", "vesicle", "endoplasmic_reticulum", "vacuole", "multivesicular_body", "membrane"),
    "TS_027": (
        "cytoplasm", "vesicle", "endoplasmic_reticulum", "nuclear_envelope", "nucleus", "vacuole", "golgi_apparatus"
    ),
    "TS_028": ("cytoplasm", "vesicle", "vacuole", "lipid_droplet", "golgi_apparatus", "multivesicular_body"),
    "TS_029": ("cytoplasm", "mitochondrion", "vesicle", "endoplasmic_reticulum", "vacuole"),
    "TS_030": ("cytoplasm", "vesicle", "endoplasmic_reticulum", "golgi_apparatus", "membrane"),
    "TS_034": (
        "cytoplasm", "mitochondrion", "vesicle", "endoplasmic_reticulum", "vacuole", "golgi_apparatus",
        "multivesicular_body", "membrane"
    ),
    "TS_037": (
        "cytoplasm", "vesicle", "nuclear_envelope", "nucleus", "vacuole", "multivesicular_body", "membrane"
    ),
    "TS_041": (
        "cytoplasm", "vesicle", "microtubule", "endoplasmic_reticulum", "vacuole", "lipid_droplet", "membrane"
    ),
    "TS_043": ("cytoplasm", "vesicle", "nuclear_envelope", "nucleus", "golgi_apparatus"),
    "TS_045": (
        "cytoplasm", "mitochondrion", "vesicle", "endoplasmic_reticulum", "nuclear_envelope", "nucleus",
        "golgi_apparatus"
    ),
}
RUNS = tuple(RUN_CLASSES.keys())
TARGETS = ("all",) + tuple(CLASSES.keys())


def _check_runs(runs):
    invalid = [run for run in runs if run not in RUN_CLASSES]
    if invalid:
        raise ValueError(f"Invalid runs {invalid}. Choose from {RUNS}.")


def _add_all_labels(zarr_path, run):
    """Merge the structures of a run into a single semantic segmentation (0 is background)."""
    import zarr
    from zarr.codecs import BloscCodec

    root = zarr.open_group(zarr_path, mode="a")
    if "all" in root:
        return

    names = [name for name in ALL_CLASSES if name in RUN_CLASSES[run]]
    shape = root["raw"].shape
    merged = np.zeros(shape, dtype="uint8")
    for name in names:
        merged[root[name][:] > 0] = ALL_CLASSES.index(name) + 1

    arr = root.create_array(
        "all", shape=shape, chunks=tuple(min(128, s) for s in shape), dtype=merged.dtype,
        compressors=BloscCodec(cname="zstd", clevel=6, shuffle="bitshuffle"),
    )
    arr[:] = merged


def get_deepict_pombe_data(
    path: Union[os.PathLike, str], runs: Union[str, List[str]] = RUNS, download: bool = False,
) -> List[str]:
    """Download the DeePiCt S. pombe data.

    The tomograms and their segmentation masks are streamed from the CryoET Data Portal and cached as zarr.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        runs: The runs (tomograms) to download. By default all runs are used.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the cached data of each run.
    """
    runs = [runs] if isinstance(runs, str) else list(runs)
    _check_runs(runs)

    os.makedirs(path, exist_ok=True)
    zarr_paths = []
    for run in runs:
        zarr_path = util.stream_from_cryo_et_portal(
            zarr_path=os.path.join(path, f"{run}.zarr"),
            dataset_id=DATASET_ID,
            run=run,
            voxel_spacing=VOXEL_SPACING,
            tomogram_folder=TOMOGRAM_FOLDER,
            annotations={name: CLASSES[name] for name in RUN_CLASSES[run]},
            download=download,
        )
        _add_all_labels(zarr_path, run)
        zarr_paths.append(zarr_path)

    return zarr_paths


def get_deepict_pombe_paths(
    path: Union[os.PathLike, str],
    target: str = "all",
    runs: Union[str, List[str]] = RUNS,
    download: bool = False,
) -> List[str]:
    """Get paths to the DeePiCt S. pombe data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        target: The segmentation target. Either 'all' for the semantic segmentation of all structures except
            the membrane, or the name of a single structure for its binary segmentation (see `CLASSES`).
        runs: The runs (tomograms) to use. By default all runs that provide the target are used.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the stored data.
    """
    if target not in TARGETS:
        raise ValueError(f"'target' must be one of {TARGETS}, got '{target}'.")
    runs = [runs] if isinstance(runs, str) else list(runs)
    _check_runs(runs)

    if target != "all":
        runs = [run for run in runs if target in RUN_CLASSES[run]]
        if not runs:
            raise ValueError(f"None of the selected runs is annotated with '{target}'.")

    return get_deepict_pombe_data(path, runs, download)


def get_deepict_pombe_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    target: Literal["all", "cytoplasm", "mitochondrion", "vesicle", "microtubule", "endoplasmic_reticulum",
                    "nuclear_envelope", "nucleus", "vacuole", "lipid_droplet", "golgi_apparatus",
                    "multivesicular_body", "membrane"] = "all",
    runs: Union[str, List[str]] = RUNS,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the dataset for structure segmentation in the DeePiCt S. pombe cryo-ET data.

    Note: Not all structures are annotated in every tomogram. For the 'all' target, structures that were not
    annotated in a tomogram are part of its background.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        target: The segmentation target. Either 'all' for the semantic segmentation of all structures except
            the membrane, or the name of a single structure for its binary segmentation (see `CLASSES`).
        runs: The runs (tomograms) to use. By default all runs that provide the target are used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_deepict_pombe_paths(path, target, runs, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key=target,
        patch_shape=patch_shape,
        **kwargs
    )


def get_deepict_pombe_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    target: Literal["all", "cytoplasm", "mitochondrion", "vesicle", "microtubule", "endoplasmic_reticulum",
                    "nuclear_envelope", "nucleus", "vacuole", "lipid_droplet", "golgi_apparatus",
                    "multivesicular_body", "membrane"] = "all",
    runs: Union[str, List[str]] = RUNS,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for structure segmentation in the DeePiCt S. pombe cryo-ET data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        target: The segmentation target. Either 'all' for the semantic segmentation of all structures except
            the membrane, or the name of a single structure for its binary segmentation (see `CLASSES`).
        runs: The runs (tomograms) to use. By default all runs that provide the target are used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_deepict_pombe_dataset(
        path, patch_shape, target=target, runs=runs, download=download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
