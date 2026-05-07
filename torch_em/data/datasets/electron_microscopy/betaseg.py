"""The BetaSeg dataset contains annotations for organelle segmentation in FIB-SEM data.

More information for this dataset is located at https://betaseg.github.io/.
And the original publication where this entire data is presented is https://arxiv.org/abs/2303.03876.
Please cite it if you use this dataset for your research.
"""

import os
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Union, Tuple, List

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://cloud.mpi-cbg.de/index.php/s/UJopHTRuh6f4wR8/download"
CHECKSUM = "4872eec0211721dc224acee319c27c4f51c190adc36004e3d5bb60dfcd67eb7b"


def get_betaseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BetaSeg dataset.

    Args:
        path: Filepath to a folder where the data will be downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(data_dir)

    zip_path = os.path.join(path, "data.zip")
    print("The BetaSeg dataset is quite large. It might take a couple of hours depending on your internet connection.")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir)

    # Group all files into h5 files.
    vol_dirs = glob(os.path.join(data_dir, "download", "*"))
    for vol_dir in tqdm(vol_dirs, desc="Preprocessing volumes"):
        # Get the image path.
        raw_path = os.path.join(vol_dir, f"{os.path.basename(vol_dir)}_source.tif")
        assert os.path.exists(raw_path), raw_path

        # Get the corresponding labels which would always exist.
        label_paths = {
            "centriole": os.path.join(vol_dir, f"{os.path.basename(vol_dir)}_centrioles.tif"),
            "golgi": os.path.join(vol_dir, f"{os.path.basename(vol_dir)}_golgi_corrected.tif"),
            "granules": os.path.join(vol_dir, f"{os.path.basename(vol_dir)}_granules.tif"),
            "membrane": os.path.join(vol_dir, f"{os.path.basename(vol_dir)}_membrane_full_mask.tif"),
            "microtubules": os.path.join(vol_dir, f"{os.path.basename(vol_dir)}_microtubules.tif"),
            "mitochondria": os.path.join(vol_dir, f"{os.path.basename(vol_dir)}_mitochondria_mask.tif"),
            "nucleus": os.path.join(vol_dir, f"{os.path.basename(vol_dir)}_nucleus_mask.tif")
        }
        for p in label_paths.values():
            assert os.path.exists(p), p

        # Load all images.
        raw = imageio.imread(raw_path)
        labels = {k: imageio.imread(v) for k, v in label_paths.items()}

        # Now, let's get all in an h5 file.
        import h5py
        vol_path = os.path.join(data_dir, Path(os.path.basename(raw_path)).with_suffix(".h5"))
        with h5py.File(vol_path, "w") as f:
            f.create_dataset("raw", data=raw, dtype=raw.dtype, compression="gzip")
            for label_key, label in labels.items():
                f.create_dataset(f"labels/{label_key}", data=label, dtype=label.dtype, compression="gzip")

    # Remove all other stuff
    shutil.rmtree(os.path.join(data_dir, "download"))

    return data_dir


def get_betaseg_paths(path: Union[os.PathLike, str], download: bool = False) -> List[str]:
    """Get filepaths to the BetaSeg data.

    Args:
        path: Filepath to a folder where the data will be downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the volumetric data.
    """
    data_dir = get_betaseg_data(path, download)
    volume_paths = glob(os.path.join(data_dir, "*.h5"))
    return volume_paths


def get_betaseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    label_choice: Union[str, List[str]],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BetaSeg dataset for organelle segmentation.

    Args:
        path: Filepath to a folder where the data will be downloaded for further processing.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of label. The choices available are: 'centriole',
            'golgi', 'granules', 'membrane', 'microtubules', 'mitochondria', 'nucleus'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_betaseg_paths(path, download)

    # Arrange the organelle choices as expecting for loading labels.
    if isinstance(label_choice, str):
        label_choices = f"labels/{label_choice}"
    else:
        label_choices = [f"labels/{organelle}" for organelle in label_choices]
        kwargs = util.update_kwargs(kwargs, "with_label_channels", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=label_choices,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs,
    )


def get_betaseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    label_choice: Union[str, List[str]],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BetaSeg dataloader for organelle segmentation.

    Args:
        path: Filepath to a folder where the data will be downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of label. The choices available are: 'centriole',
            'golgi', 'granules', 'membrane', 'microtubules', 'mitochondria', 'nucleus'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_betaseg_dataset(path, patch_shape, label_choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
