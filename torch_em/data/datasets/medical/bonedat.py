"""BoneDat is a dataset for the segmentation of the pelvis, sacrum and L4/L5 vertebrae in lumbopelvic CT.

The dataset consists of 278 anonymized human lumbopelvic CT scans (ages 16-91, balanced by sex), together with
multi-label segmentation masks that delineate the individual pelvic bones, the sacrum and the L4/L5 vertebrae.
The masks were generated with the Biomedisa algorithm and refined and validated by experts.

The dataset is hosted on Zenodo as 8 linked records. The main record (https://doi.org/10.5281/zenodo.15189761)
only holds supplementary results and metadata from the associated publication, not usable image data. The raw
CT scans and segmentation masks are distributed across 7 records that together form a single split rar archive
(BoneDat.part1.rar - BoneDat.part7.rar, https://doi.org/10.5281/zenodo.15188359 and follow-up DOIs). Most of
this archive (parts 1-6, about 290 GB) holds registration and template outputs (deformation fields, warped
volumes, meshes) that are not exposed by this module. The raw CT scans and segmentation masks used here are
both fully contained within part 7 alone (about 27 GB), so this module only downloads that part.

All 8 records are licensed under CC-BY-4.0. Note that the article text itself is under a separate, more
restrictive Springer Nature license, but this does not apply to the dataset files, which are CC-BY-4.0.

This dataset is from the publication https://doi.org/10.1038/s41597-025-05161-y. Please cite it if you use this
dataset for your research.
"""

import os
import shutil
import subprocess
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/15189605/files/BoneDat.part7.rar?download=1"
CHECKSUM = "f27bf0813df07c1cceff822246e2793bb03ca74e10f6e7e87861e19b9bdafb85"


def _extract_bonedat(rar_path, data_dir):
    if shutil.which("7z") is None:
        raise RuntimeError(
            "Need the 'p7zip' CLI to extract this archive. You can install it via 'conda install -c conda-forge p7zip'."  # noqa
        )

    # The rar archive is part of a 7-part split archive (this is part 7). Extracting it in isolation (without
    # the preceding 6 parts) makes '7z' report a non-fatal 'Headers Error' for the missing volume chain, but the
    # files that are fully contained within this part (i.e. everything under 'raw/' and 'derived/segmentation/')
    # are still extracted correctly. We therefore do not check the subprocess return code here and instead
    # verify the presence of the expected output further below.
    subprocess.run(["7z", "x", f"-o{data_dir}", "-y", rar_path, "raw/*", "derived/segmentation/*"])


def get_bonedat_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BoneDat dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(os.path.join(data_dir, "raw")) and os.path.exists(os.path.join(data_dir, "derived")):
        return data_dir

    os.makedirs(path, exist_ok=True)

    rar_path = os.path.join(path, "BoneDat.part7.rar")
    util.download_source(path=rar_path, url=URL, download=download, checksum=CHECKSUM)

    os.makedirs(data_dir, exist_ok=True)
    _extract_bonedat(rar_path, data_dir)

    if not os.path.exists(os.path.join(data_dir, "raw")) or not os.path.exists(os.path.join(data_dir, "derived")):
        raise RuntimeError(f"Extraction seems to have failed: could not find the expected data at '{data_dir}'.")

    os.remove(rar_path)

    return data_dir


def get_bonedat_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the BoneDat data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_bonedat_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "raw", "*", "original.nii.gz")))
    label_paths = natsorted(glob(os.path.join(data_dir, "derived", "segmentation", "*", "mask.nii.gz")))

    if len(raw_paths) == 0 or len(raw_paths) != len(label_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return raw_paths, label_paths


def get_bonedat_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BoneDat dataset for pelvis, sacrum and L4/L5 vertebra segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_bonedat_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_bonedat_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BoneDat dataloader for pelvis, sacrum and L4/L5 vertebra segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bonedat_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
