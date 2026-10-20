"""This dataset provides organelle segmentation masks for cryo-ET tomograms of
Chlamydomonas reinhardtii cells, prepared by cryo-plasmaFIB milling.

The data is hosted on the CryoET Data Portal as dataset 10301
(https://cryoetdataportal.czscience.com/datasets/10301, EMPIAR-11756). This loader covers three of its runs:
- 17072022_BrnoKrios_Arctis_p3ar_grid_Position_76 (https://cryoetdataportal.czscience.com/runs/14082)
- 17072022_BrnoKrios_Arctis_p3ar_grid_Position_101 (https://cryoetdataportal.czscience.com/runs/14079)
- 27042022_BrnoKrios_Arctis_grid9_hGIS_Position_44 (https://cryoetdataportal.czscience.com/runs/14085)

NOTE: The expert-curated ground truth of these runs consists of point annotations (ribosomes, microtubules,
RubisCO) only. All dense masks used here were predicted automatically with Ais / Pom
(`ground_truth_status=false`, `method_type=automated`) and have not been proofread, so they are
pseudo-labels of varying quality. Only structures with masks in at least two of the three runs are provided.

The data is described in https://doi.org/10.7554/eLife.98552 and https://doi.org/10.1101/2025.01.16.633326.
Please cite them if you use this dataset in your research.
"""

import os
from typing import List, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


DATASET_ID = 10301
VOXEL_SPACING = "VoxelSpacing7.840"
TOMOGRAM_FOLDER = "100"

# The annotation folder of each mask differs between the runs.
RUN_LABELS = {
    "17072022_BrnoKrios_Arctis_p3ar_grid_Position_76": {
        "mitochondrion": "114", "lipid_droplet": "132", "cytoplasmic_vesicle": "125", "pyrenoid": "130",
        "starch_grain": "138", "nucleoplasm": "127", "endoplasmic_reticulum": "133", "golgi_apparatus": "135",
        "cell_wall": "136", "chloroplast_stroma": "131", "cytoplasm": "113",
    },
    "17072022_BrnoKrios_Arctis_p3ar_grid_Position_101": {
        "mitochondrion": "120", "lipid_droplet": "125", "cytoplasmic_vesicle": "129", "pyrenoid": "118",
        "starch_grain": "132", "chloroplast_stroma": "124",
    },
    "27042022_BrnoKrios_Arctis_grid9_hGIS_Position_44": {
        "mitochondrion": "123", "lipid_droplet": "130", "cytoplasmic_vesicle": "122", "pyrenoid": "128",
        "starch_grain": "136", "nucleoplasm": "125", "endoplasmic_reticulum": "131", "golgi_apparatus": "133",
        "cell_wall": "134", "cytoplasm": "110",
    },
}
RUNS = list(RUN_LABELS.keys())
TARGETS = (
    "mitochondrion", "lipid_droplet", "cytoplasmic_vesicle", "pyrenoid", "starch_grain", "nucleoplasm",
    "endoplasmic_reticulum", "golgi_apparatus", "cell_wall", "chloroplast_stroma", "cytoplasm",
)


def get_chlamydomonas_cryoet_data(
    path: Union[os.PathLike, str], runs: Union[str, List[str]] = RUNS, download: bool = False,
) -> List[str]:
    """Download the Chlamydomonas cryo-ET data.

    The tomograms and their segmentation masks are streamed from the CryoET Data Portal and cached as zarr.
    Requires zarr (v3) and s3fs.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        runs: The runs (tomograms) to download. By default all runs are used.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the cached data of each run.
    """
    runs = [runs] if isinstance(runs, str) else list(runs)
    invalid = [run for run in runs if run not in RUN_LABELS]
    if invalid:
        raise ValueError(f"Invalid runs {invalid}. Choose from {RUNS}.")

    os.makedirs(path, exist_ok=True)
    return [
        util.stream_from_cryo_et_portal(
            zarr_path=os.path.join(path, f"{run}.zarr"),
            dataset_id=DATASET_ID,
            run=run,
            voxel_spacing=VOXEL_SPACING,
            tomogram_folder=TOMOGRAM_FOLDER,
            annotations={name: folder for name, folder in RUN_LABELS[run].items()},
            download=download,
        ) for run in runs
    ]


def get_chlamydomonas_cryoet_paths(
    path: Union[os.PathLike, str],
    target: str = "mitochondrion",
    runs: Union[str, List[str]] = RUNS,
    download: bool = False,
) -> List[str]:
    """Get paths to the Chlamydomonas cryo-ET data that provide the requested target.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        target: The segmentation target, one of `TARGETS`.
        runs: The runs (tomograms) to use. By default all runs providing the target are used.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths to the cached data.
    """
    if target not in TARGETS:
        raise ValueError(f"'target' must be one of {TARGETS}, got '{target}'.")
    runs = [runs] if isinstance(runs, str) else list(runs)
    runs = [run for run in runs if run in RUN_LABELS and target in RUN_LABELS[run]]
    if not runs:
        raise ValueError(f"None of the selected runs provides the target '{target}'.")
    return get_chlamydomonas_cryoet_data(path, runs, download)


def get_chlamydomonas_cryoet_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    target: str = "mitochondrion",
    runs: Union[str, List[str]] = RUNS,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the dataset for organelle segmentation in Chlamydomonas cryo-ET data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        target: The segmentation target, one of `TARGETS`.
        runs: The runs (tomograms) to use. By default all runs providing the target are used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert len(patch_shape) == 3

    paths = get_chlamydomonas_cryoet_paths(path, target, runs, download)

    kwargs = util.update_kwargs(kwargs, "is_seg_dataset", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key=target,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_chlamydomonas_cryoet_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    target: str = "mitochondrion",
    runs: Union[str, List[str]] = RUNS,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the DataLoader for organelle segmentation in Chlamydomonas cryo-ET data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape (z, y, x) to use for training.
        batch_size: The batch size for training.
        target: The segmentation target, one of `TARGETS`.
        runs: The runs (tomograms) to use. By default all runs providing the target are used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`
            or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_chlamydomonas_cryoet_dataset(
        path, patch_shape, target=target, runs=runs, download=download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
