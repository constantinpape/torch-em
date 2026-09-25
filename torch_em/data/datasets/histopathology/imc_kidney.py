"""This dataset contains cell instance segmentation annotations for imaging mass cytometry (IMC)
of human kidney biopsies with acute interstitial nephritis (AIN), acute tubular injury (ATI), and
histologically normal reference tissue.

The data is from the publication "Spatial analysis reveals cellular microenvironments and
mechanisms of inflammation and injury in acute interstitial nephritis" and hosted on the
BioImage Archive at https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD3694. It is
available under the CC0 license. Please cite it if you use this dataset in your research.

This loader covers the Discovery cohort (23 experimental batches, 325 acquisitions), which is
provided as processed multichannel TIFF stacks (35-marker antibody panel) with matching per-cell
instance segmentation masks generated with Mesmer. The BioImage Archive record also hosts a
Validation cohort of raw Hyperion text exports without segmentation masks, which is out of scope
for this loader.
"""

import os
from glob import glob
from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/694/S-BIAD3694/Files"

BATCHES = tuple(f"Batch{i}" for i in range(1, 24))


def _get_manifest(path, download):
    # NOTE: The per-batch 'images.csv' manifests list acquisitions that were planned but not all of
    # them were actually deposited (e.g. excluded ROIs). The top-level file list only contains the
    # acquisitions that were actually uploaded, so we rely on it to determine the real image / mask pairs.
    manifest_path = os.path.join(path, "bia_filelist_all.tsv")
    os.makedirs(path, exist_ok=True)
    util.download_source(manifest_path, f"{BASE_URL}/bia_filelist_all.tsv", download, checksum=None)
    manifest = pd.read_csv(manifest_path, sep="\t")
    return manifest[manifest["cohort"] == "discovery"]


def get_imc_kidney_data(
    path: Union[os.PathLike, str],
    batches: Optional[Sequence[str]] = None,
    download: bool = False,
) -> str:
    """Download the IMC kidney (AIN) Discovery cohort data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batches: The batch names to prepare. By default all 23 batches are prepared, which
            requires downloading several tens of gigabytes of data.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the data is stored.
    """
    if batches is None:
        batches = BATCHES
    else:
        invalid = sorted(set(batches) - set(BATCHES))
        if invalid:
            raise ValueError(f"Invalid batch name(s) {invalid}. Choose from {BATCHES}.")

    os.makedirs(path, exist_ok=True)
    manifest = _get_manifest(path, download)
    manifest = manifest[manifest["batch"].isin(batches) & manifest["file_role"].isin(["image", "segmentation_mask"])]

    for file_path in manifest["Files"]:
        local_path = os.path.join(path, file_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        util.download_source(local_path, f"{BASE_URL}/{file_path}", download, checksum=None)

    return path


def get_imc_kidney_paths(
    path: Union[os.PathLike, str],
    batches: Optional[Sequence[str]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the IMC kidney (AIN) images and cell instance segmentation masks.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batches: The batch names to load. By default all 23 batches are loaded.
        download: Whether to download the data if it is not present.

    Returns:
        The image paths and corresponding mask paths.
    """
    root = get_imc_kidney_data(path, batches, download)
    if batches is None:
        batches = BATCHES

    raw_paths, label_paths = [], []
    for batch in batches:
        raw_paths.extend(sorted(glob(os.path.join(root, "Discovery", batch, "img", "*.tiff"))))
        label_paths.extend(sorted(glob(os.path.join(root, "Discovery", batch, "masks", "*.tiff"))))

    missing_paths = [p for p in raw_paths + label_paths if not os.path.exists(p)]
    if missing_paths:
        raise RuntimeError(f"Could not find {len(missing_paths)} IMC kidney files.")

    return raw_paths, label_paths


def get_imc_kidney_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batches: Optional[Sequence[str]] = None,
    download: bool = False,
    resize_inputs: bool = False,
    **kwargs
) -> Dataset:
    """Get the IMC kidney (AIN) dataset for cell instance segmentation in imaging mass cytometry.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        batches: The batch names to load. By default all 23 batches are loaded.
        download: Whether to download the data if it is not present.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The IMC kidney patch shape must be two-dimensional, got {patch_shape}.")

    raw_paths, label_paths = get_imc_kidney_paths(path, batches, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_imc_kidney_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    batches: Optional[Sequence[str]] = None,
    download: bool = False,
    resize_inputs: bool = False,
    **kwargs
) -> DataLoader:
    """Get the IMC kidney (AIN) dataloader for cell instance segmentation in imaging mass cytometry.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        batch_size: The batch size for training.
        batches: The batch names to load. By default all 23 batches are loaded.
        download: Whether to download the data if it is not present.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for
            the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_imc_kidney_dataset(
        path, patch_shape, batches=batches, download=download, resize_inputs=resize_inputs, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
