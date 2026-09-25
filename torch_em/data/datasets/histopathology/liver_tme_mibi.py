"""This dataset contains cell instance segmentation annotations for multiplexed ion beam imaging
(MIBI) of syngeneic orthotopic murine liver cancer models, covering intrahepatic cholangiocarcinoma
(iCCA) and hepatocellular carcinoma (HCC) driven by Trp53del / KrasG12D mutations, as well as
FGFR2 fusion-driven iCCA.

The data is from the study "Multidimensional spatial profiling of the tumor microenvironment in
syngeneic orthotopic murine liver cancer models", hosted on the BioImage Archive at
https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2557. Please cite it if you use this
dataset in your research.

This loader covers the 562 fields of view (FOVs). Each FOV is a 1024x1024 crop with 30 protein
marker channels (covering cancer, immune and stromal lineages) and a matching per-cell instance
segmentation mask, generated with Cellpose on a nuclear (histone H3) and membrane marker composite
and filtered to high-confidence single cells. NOTE: The full set of FOVs is very large
(the 30-channel images alone total about 140 GB); use the `fovs` argument to restrict which ones
are downloaded and preprocessed.

On first use, each requested FOV is converted into a single HDF5 file with the following layout:
    - 'raw/all': the (30, H, W) stack of all channels.
    - 'raw/channels/<channel>': each individual channel (H, W), see `CHANNELS` for the full list.
    - 'labels/instances': the instance segmentation.
"""

import os
from glob import glob
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/557/S-BIAD2557/Files/spatial_murine_iCCAvsHCC"
MANIFEST_URL = f"{BASE_URL}/MIBI_images_file_list.json"

CHANNELS = (
    "ATP5A", "B220", "cCASP3", "CD11c", "CD163", "CD31", "CD3e", "CD45", "CD4", "COL1A1", "CTLA4",
    "CytC", "F4_80", "G6PD", "GLUT1", "HH3", "HNF4a", "Ki67", "LDH", "Ly6G", "mCD11b", "mCD44",
    "mCD8", "mFoxP3", "mPD_1", "mPD_L1", "NaKATPase", "PanCK", "SMA", "Vimentin",
)


def _get_manifest(path, download):
    import requests

    manifest_path = os.path.join(path, "MIBI_images_file_list.json")
    if not os.path.exists(manifest_path):
        if not download:
            raise RuntimeError(f"Cannot find the manifest at {manifest_path}, but download was set to False")
        response = requests.get(MANIFEST_URL)
        response.raise_for_status()
        with open(manifest_path, "wb") as f:
            f.write(response.content)

    import json
    with open(manifest_path) as f:
        manifest = json.load(f)

    return sorted(os.path.basename(entry["path"]) for entry in manifest)


def _convert_fov(fov_id, output_path, download):
    import tifffile

    tmp_path = output_path + ".tmp"

    channel_stack = []
    for channel in CHANNELS:
        channel_url = f"{BASE_URL}/image_data/{fov_id}/{channel}.tiff"
        channel_path = tmp_path + f".{channel}.tiff"
        util.download_source(channel_path, channel_url, download, checksum=None)
        channel_stack.append(tifffile.imread(channel_path))
        os.remove(channel_path)
    raw = np.stack(channel_stack, axis=0)

    mask_url = f"{BASE_URL}/segmentation/cleaned_mask/{fov_id}_cleaned_mask.tiff"
    mask_path = tmp_path + ".mask.tiff"
    util.download_source(mask_path, mask_url, download, checksum=None)
    instances = tifffile.imread(mask_path)
    os.remove(mask_path)

    import h5py
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw/all", data=raw, compression="gzip", chunks=(len(CHANNELS), 512, 512))
        for i, name in enumerate(CHANNELS):
            f.create_dataset(f"raw/channels/{name}", data=raw[i], compression="gzip")
        f.create_dataset("labels/instances", data=instances, compression="gzip")

    os.replace(tmp_path, output_path)


def get_liver_tme_mibi_data(
    path: Union[os.PathLike, str], fovs: Optional[Sequence[str]] = None, download: bool = False,
) -> str:
    """Download and preprocess the murine liver TME MIBI data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        fovs: The field of view (FOV) ids to prepare. By default all 562 FOVs are prepared, which
            requires downloading about 140 GB of data. See `MANIFEST_URL` for the full list of ids.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the preprocessed data is stored.
    """
    os.makedirs(path, exist_ok=True)
    valid_fovs = _get_manifest(path, download)

    if fovs is None:
        fovs = valid_fovs
    else:
        invalid = sorted(set(fovs) - set(valid_fovs))
        if invalid:
            raise ValueError(f"Invalid FOV id(s) {invalid}.")

    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    for fov_id in fovs:
        output_path = os.path.join(preprocessed_dir, f"{fov_id}.h5")
        if os.path.exists(output_path):
            continue
        _convert_fov(fov_id, output_path, download)

    return preprocessed_dir


def get_liver_tme_mibi_paths(
    path: Union[os.PathLike, str], fovs: Optional[Sequence[str]] = None, download: bool = False,
) -> List[str]:
    """Get paths to the preprocessed murine liver TME MIBI data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        fovs: The field of view (FOV) ids to load. By default all 562 FOVs are loaded.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the preprocessed HDF5 files.
    """
    preprocessed_dir = get_liver_tme_mibi_data(path, fovs, download)
    if fovs is None:
        paths = sorted(glob(os.path.join(preprocessed_dir, "*.h5")))
    else:
        paths = [os.path.join(preprocessed_dir, f"{fov_id}.h5") for fov_id in fovs]

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"Could not find the data at {missing}.")

    return paths


def get_liver_tme_mibi_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    fovs: Optional[Sequence[str]] = None,
    channel: str = "all",
    download: bool = False,
    resize_inputs: bool = False,
    **kwargs
) -> Dataset:
    """Get the murine liver TME MIBI dataset for cell instance segmentation in multiplexed
    ion beam images of the liver cancer tumor microenvironment.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        fovs: The field of view (FOV) ids to load. By default all 562 FOVs are loaded.
        channel: The raw input. Either 'all' for the full (30, H, W) channel stack, or the name of
            a single channel, see `CHANNELS` for the full list, e.g. 'PanCK'.
        download: Whether to download the data if it is not present.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if channel == "all":
        raw_key, with_channels = "raw/all", True
    elif channel in CHANNELS:
        raw_key, with_channels = f"raw/channels/{channel}", False
    else:
        raise ValueError(f"'{channel}' is not a valid channel. Choose 'all' or one of {CHANNELS}.")

    paths = get_liver_tme_mibi_paths(path, fovs, download)

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


def get_liver_tme_mibi_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    fovs: Optional[Sequence[str]] = None,
    channel: str = "all",
    download: bool = False,
    resize_inputs: bool = False,
    **kwargs
) -> DataLoader:
    """Get the murine liver TME MIBI dataloader for cell instance segmentation in multiplexed
    ion beam images of the liver cancer tumor microenvironment.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        fovs: The field of view (FOV) ids to load. By default all 562 FOVs are loaded.
        channel: The raw input. Either 'all' for the full (30, H, W) channel stack, or the name of
            a single channel, see `CHANNELS` for the full list, e.g. 'PanCK'.
        download: Whether to download the data if it is not present.
        resize_inputs: Whether to resize the input images.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the
            PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_liver_tme_mibi_dataset(
        path, patch_shape, fovs=fovs, channel=channel, download=download, resize_inputs=resize_inputs, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
