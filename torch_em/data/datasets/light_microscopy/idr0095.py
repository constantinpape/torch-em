"""The IDR0095 dataset (idr0095-ali-asymmetry) contains fluorescence microscopy
images of Escherichia coli cells from three experiments studying regulatory asymmetry
in transcription factor autoregulatory gene networks.

Each acquisition contains three imaging channels:
- Phase contrast (channel 0): cell morphology — used as raw input for segmentation
- mCherry (channel 1): transcription factor gene expression level
- YFP (channel 2): downstream target gene expression level

The Phase channel images are extracted from Nikon ND2 files and paired with
hand-segmented cell instance masks. Reading ND2 files requires the `nd2` package:
    pip install nd2

Data is hosted on EBI FTP: ftp.ebi.ac.uk/pub/databases/IDR/idr0095-ali-asymmetry/
The dataset accession on IDR is idr0095.

This dataset is from the following publication:
- Ali et al. (2020): https://doi.org/10.7554/eLife.56517
Please cite it if you use this dataset in your research.
"""

import ftplib
import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Tuple, Union

import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


FTP_HOST = "ftp.ebi.ac.uk"
FTP_BASE = "/pub/databases/IDR/idr0095-ali-asymmetry/20200804-ftp"

EXPERIMENT_DIRS = {
    "A": "Experiment_A_Figure3",
    "B": "Experiment_B_Figure4",
    "C": "Experiment_C_Figure5",
}


def _ftp_download_recursive(ftp: ftplib.FTP, remote_dir: str, local_dir: str) -> None:
    os.makedirs(local_dir, exist_ok=True)
    ftp.cwd(remote_dir)

    entries = []
    ftp.retrlines("LIST", entries.append)

    for entry in entries:
        parts = entry.split()
        name = parts[-1]
        is_dir = entry.startswith("d")
        local_path = os.path.join(local_dir, name)

        if is_dir:
            _ftp_download_recursive(ftp, f"{remote_dir}/{name}", local_path)
            ftp.cwd(remote_dir)  # return to parent after recursion
        else:
            if not os.path.exists(local_path):
                with open(local_path, "wb") as f:
                    ftp.retrbinary(f"RETR {name}", f.write)


def _get_phase_channel_index(nd2_file) -> int:
    """Return the index of the Phase channel, defaulting to 0."""
    try:
        names = [ch.channel.name.lower() for ch in nd2_file.metadata.channels]
        for i, name in enumerate(names):
            if "phase" in name or "bf" in name or "trans" in name:
                return i
    except Exception:
        pass
    return 0


def _extract_phase_tiffs(exp_dir: str, experiment: str) -> None:
    """Extract Phase channel frames from all ND2 files in an experiment directory."""
    try:
        import nd2
    except ImportError:
        raise ImportError(
            "The 'nd2' package is required to read ND2 files from IDR0095. "
            "Please install it with: pip install nd2"
        )

    nd2_files = natsorted(glob(os.path.join(exp_dir, "**", "*.nd2"), recursive=True))
    if not nd2_files:
        raise RuntimeError(f"No ND2 files found in {exp_dir}.")

    for nd2_path in tqdm(nd2_files, desc=f"Extracting Phase TIFFs (Experiment {experiment})"):
        condition = os.path.splitext(os.path.basename(nd2_path))[0]
        mask_dir = os.path.join(os.path.dirname(nd2_path), condition)

        if not os.path.isdir(mask_dir):
            continue

        mask_paths = natsorted(glob(os.path.join(mask_dir, "*-Mask.tif")))
        if not mask_paths:
            continue

        phase_paths = [p.replace("-Mask.tif", "-Phase.tif") for p in mask_paths]
        if all(os.path.exists(p) for p in phase_paths):
            continue  # already extracted

        try:
            with nd2.ND2File(nd2_path) as f:
                phase_idx = _get_phase_channel_index(f)
                arr = f.asarray()  # shape varies by acquisition settings
        except Exception as e:
            print(f"Warning: skipping {nd2_path} — could not read ND2 file: {e}")
            continue

        # Normalize to (N_fields, N_channels, H, W).
        # nd2.asarray() may return (P, C, Y, X), (C, Y, X), (Y, X), etc.
        if arr.ndim == 2:
            arr = arr[np.newaxis, np.newaxis]  # (1, 1, H, W)
        elif arr.ndim == 3:
            arr = arr[:, np.newaxis]  # (P, 1, H, W) or (C, H, W)?
        # If 4-D, assume (P, C, H, W) — standard nd2 layout for multi-position/channel.

        n_frames = arr.shape[0]

        for i, (mask_path, phase_path) in enumerate(zip(mask_paths, phase_paths)):
            if os.path.exists(phase_path):
                continue
            frame_idx = min(i, n_frames - 1)
            frame = arr[frame_idx, phase_idx] if arr.ndim == 4 else arr[frame_idx, 0]
            imageio.imwrite(phase_path, frame.astype(np.uint16))


def get_idr0095_data(
    path: Union[os.PathLike, str],
    experiment: Literal["A", "B", "C"] = "A",
    download: bool = False,
) -> str:
    """Download the IDR0095 dataset from EBI FTP and extract Phase channel TIFFs.

    NOTE: This dataset is large — Experiment A is ~6 GB, B ~9 GB, C ~18 GB.
    Raw images are in Nikon ND2 format; the `nd2` package (pip install nd2)
    is required to extract the Phase channel TIFFs on first use.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        experiment: The experiment to download. One of 'A', 'B', or 'C',
            corresponding to Figures 3, 4, and 5 of Ali et al. (2020).
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the data directory containing the experiment folder.
    """
    assert experiment in EXPERIMENT_DIRS, \
        f"'{experiment}' is not a valid experiment. Choose from {list(EXPERIMENT_DIRS)}."

    data_dir = os.path.join(path, "idr0095")
    exp_dir = os.path.join(data_dir, EXPERIMENT_DIRS[experiment])

    if not download and not os.path.exists(exp_dir):
        raise RuntimeError(
            f"IDR0095 experiment {experiment} not found at {exp_dir}. "
            "Set download=True to download from EBI FTP."
        )

    if download:
        os.makedirs(data_dir, exist_ok=True)
        print(f"Connecting to {FTP_HOST} to download IDR0095 Experiment {experiment} ...")
        print("This may take a long time depending on experiment size (~6–18 GB).")
        with ftplib.FTP(FTP_HOST) as ftp:
            ftp.login()  # anonymous login
            # _ftp_download_recursive skips files that already exist, safe to re-run.
            _ftp_download_recursive(ftp, f"{FTP_BASE}/{EXPERIMENT_DIRS[experiment]}", exp_dir)

    _extract_phase_tiffs(exp_dir, experiment)
    return data_dir


def get_idr0095_paths(
    path: Union[os.PathLike, str],
    experiment: Literal["A", "B", "C"] = "A",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to IDR0095 Phase-channel images and cell segmentation masks.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        experiment: The experiment to use. One of 'A', 'B', or 'C'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the Phase-channel TIFF images.
        List of filepaths to the instance segmentation mask TIFFs.
    """
    data_dir = get_idr0095_data(path, experiment, download)
    exp_dir = os.path.join(data_dir, EXPERIMENT_DIRS[experiment])

    mask_paths = natsorted(glob(os.path.join(exp_dir, "**", "*-Mask.tif"), recursive=True))
    pairs = [
        (p.replace("-Mask.tif", "-Phase.tif"), p)
        for p in mask_paths
        if os.path.exists(p.replace("-Mask.tif", "-Phase.tif"))
    ]

    if not pairs:
        raise RuntimeError(
            f"No Phase TIFFs found in {exp_dir}. "
            "Ensure the dataset was downloaded and nd2 is installed for Phase extraction."
        )

    raw_paths, mask_paths = zip(*pairs)
    return list(raw_paths), list(mask_paths)


def get_idr0095_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    experiment: Literal["A", "B", "C"] = "A",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the IDR0095 dataset for E. coli phase-contrast cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        experiment: The experiment to use. One of 'A', 'B', or 'C'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_idr0095_paths(path, experiment, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs,
    )


def get_idr0095_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    experiment: Literal["A", "B", "C"] = "A",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the IDR0095 dataloader for E. coli phase-contrast cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        experiment: The experiment to use. One of 'A', 'B', or 'C'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_idr0095_dataset(path, patch_shape, experiment, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
