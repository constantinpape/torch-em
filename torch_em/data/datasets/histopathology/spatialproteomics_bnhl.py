"""This dataset contains cell instance segmentation annotations for highly multiplexed
immunofluorescence imaging of reactive lymph nodes and various B cell Non-Hodgkin lymphomas (BNHL),
acquired with the Akoya Phenocycler.

The data is from the publication https://doi.org/10.1038/s41592-026-03155-1 and hosted on the
BioImage Archive at https://www.ebi.ac.uk/biostudies/studies/S-BIAD2100. Please cite it if you use
this dataset in your research.

This loader covers the 250 tissue microarray (TMA) cores. Each core is a 3000x3000 pixel crop with
56 immunofluorescence channels and a matching per-cell instance segmentation mask, generated with
cellpose on the DAPI channel. The BioImage Archive record also hosts 5 whole-slide images
(24-63 GB each), which are out of scope for this loader.

On first use, each TMA core is converted into a single HDF5 file with the following layout:
    - 'raw/all': the (56, H, W) stack of all channels.
    - 'raw/channels/<channel>': each individual channel (H, W), see `CHANNELS` for the full list.
    - 'labels/instances': the instance segmentation.
"""

import os
from glob import glob
from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/100/S-BIAD2100/Files/data_for_publication"
MANIFEST_URL = f"{BASE_URL}/derived_data/tma_sample_overview.csv"
MANIFEST_CHECKSUM = "c2bb69d29af94c09cb7cd5b3b10e3acfdffb6beea1d7e8dd46b93cc30f55b0f8"

CHANNELS = (
    "DAPI", "Helios", "CD10", "TCF7/TCF1", "PD-L1", "BCL-6", "FOXP3", "CD69", "Perforin", "CD19",
    "LAG3", "CD21", "CD62L", "c-myc", "CD138", "CD15", "BCL-2", "CD56", "IKZF3", "CD25", "NOXA",
    "Tim3", "Serpin B9", "Podoplanin", "CD38", "SPARC", "ICOS", "CXCR5", "CD163", "FADD", "p53",
    "Collagen IV", "CD4", "CD7", "Kappa", "CD20", "CD34", "PAX5", "PD-1", "CD45RA", "CD11b",
    "Lambda", "CD57", "CD11c", "CD90", "HLA DR", "CD68", "CD31", "CD45", "CD3", "Cytokeratin",
    "CD45RO", "CD8", "Granzyme B", "CD79a", "Ki-67",
)


def _sanitize_channel(name):
    return name.replace("/", "-")


def _get_manifest(path, download):
    manifest_path = os.path.join(path, "tma_sample_overview.csv")
    util.download_source(manifest_path, MANIFEST_URL, download, MANIFEST_CHECKSUM)
    return pd.read_csv(manifest_path, index_col=0)


def _convert_sample(zip_path, output_path):
    import h5py
    import zarr

    store = zarr.storage.ZipStore(zip_path, mode="r")
    group = zarr.open_group(store=store, mode="r")
    channels = [str(c) for c in group["channels"][:]]
    if channels != list(CHANNELS):
        raise RuntimeError(f"Unexpected channel order in {zip_path}.")

    raw = group["_image_raw"][:]
    instances = group["_segmentation"][:]
    store.close()

    tmp_path = output_path + ".tmp"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw/all", data=raw, compression="gzip", chunks=(len(CHANNELS), 512, 512))
        for i, name in enumerate(CHANNELS):
            f.create_dataset(f"raw/channels/{_sanitize_channel(name)}", data=raw[i], compression="gzip")
        f.create_dataset("labels/instances", data=instances, compression="gzip")

    os.replace(tmp_path, output_path)


def get_spatialproteomics_bnhl_data(
    path: Union[os.PathLike, str],
    samples: Optional[Sequence[str]] = None,
    download: bool = False,
) -> str:
    """Download and preprocess the spatialproteomics BNHL tissue microarray (TMA) data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        samples: The TMA sample ids to prepare. By default all 250 TMA cores are prepared, which
            requires downloading about 60 GB of data.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the preprocessed data is stored.
    """
    os.makedirs(path, exist_ok=True)
    manifest = _get_manifest(path, download)
    valid_samples = set(manifest["sample_id"])

    if samples is None:
        samples = sorted(valid_samples)
    else:
        invalid = sorted(set(samples) - valid_samples)
        if invalid:
            raise ValueError(f"Invalid sample id(s) {invalid}.")

    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    zip_dir = os.path.join(path, "tmas")
    os.makedirs(zip_dir, exist_ok=True)

    for sample_id in samples:
        output_path = os.path.join(preprocessed_dir, f"{sample_id}.h5")
        if os.path.exists(output_path):
            continue

        zip_path = os.path.join(zip_dir, f"{sample_id}.zarr.zip")
        util.download_source(zip_path, f"{BASE_URL}/tmas/{sample_id}.zarr.zip", download, checksum=None)
        _convert_sample(zip_path, output_path)

    return preprocessed_dir


def get_spatialproteomics_bnhl_paths(
    path: Union[os.PathLike, str],
    samples: Optional[Sequence[str]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the preprocessed spatialproteomics BNHL TMA data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        samples: The TMA sample ids to load. By default all 250 TMA cores are loaded.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the preprocessed HDF5 files.
    """
    preprocessed_dir = get_spatialproteomics_bnhl_data(path, samples, download)
    if samples is None:
        paths = sorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    else:
        paths = [os.path.join(preprocessed_dir, f"{sample_id}.h5") for sample_id in samples]

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"Could not find the data at {missing}.")

    return paths


def get_spatialproteomics_bnhl_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    samples: Optional[Sequence[str]] = None,
    channel: str = "all",
    download: bool = False,
    resize_inputs: bool = False,
    **kwargs
) -> Dataset:
    """Get the spatialproteomics BNHL dataset for cell instance segmentation in multiplexed
    immunofluorescence images of B cell Non-Hodgkin lymphomas.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        samples: The TMA sample ids to load. By default all 250 TMA cores are loaded.
        channel: The raw input. Either 'all' for the full (56, H, W) channel stack, or the name of
            a single channel, see `CHANNELS` for the full list, e.g. 'CD20'.
        download: Whether to download the data if it is not present.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if channel == "all":
        raw_key, with_channels = "raw/all", True
    elif channel in CHANNELS:
        raw_key, with_channels = f"raw/channels/{_sanitize_channel(channel)}", False
    else:
        raise ValueError(f"'{channel}' is not a valid channel. Choose 'all' or one of {CHANNELS}.")

    paths = get_spatialproteomics_bnhl_paths(path, samples, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key=raw_key,
        label_paths=paths,
        label_key="labels/instances",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        with_channels=with_channels,
        ndim=2,
        **kwargs
    )


def get_spatialproteomics_bnhl_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    samples: Optional[Sequence[str]] = None,
    channel: str = "all",
    download: bool = False,
    resize_inputs: bool = False,
    **kwargs
) -> DataLoader:
    """Get the spatialproteomics BNHL dataloader for cell instance segmentation in multiplexed
    immunofluorescence images of B cell Non-Hodgkin lymphomas.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        samples: The TMA sample ids to load. By default all 250 TMA cores are loaded.
        channel: The raw input. Either 'all' for the full (56, H, W) channel stack, or the name of
            a single channel, see `CHANNELS` for the full list, e.g. 'CD20'.
        download: Whether to download the data if it is not present.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the
            PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_spatialproteomics_bnhl_dataset(
        path, patch_shape, samples=samples, channel=channel, download=download,
        resize_inputs=resize_inputs, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
