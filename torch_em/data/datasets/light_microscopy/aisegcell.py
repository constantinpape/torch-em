"""The aiSEGcell dataset contains annotations for nucleus segmentation in
paired brightfield and fluorescence images.

The dataset collection is located at https://www.research-collection.ethz.ch/handle/20.500.11850/679085.
This dataset is from the publication https://doi.org/10.1371/journal.pcbi.1012361.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import List, Union, Tuple, Literal
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://libdrive.ethz.ch/index.php/s/VoF2SYkbLY8izjh/download"
CHECKSUM = "f9115ee6b71e7c4364b83f7d7f8b66dce5b778344070bddb6a8f0e5086ca5de9"


def _process_each_image(args):
    import h5py

    bpath, npath, gpath, data_dir = args

    path_parents = Path(bpath).parents
    split = path_parents[1].name.split("_")[-1]
    dname = path_parents[2].name

    neu_dir = os.path.join(data_dir, split, dname)
    os.makedirs(neu_dir, exist_ok=True)

    fpath = os.path.join(neu_dir, f"{Path(bpath).stem}.h5")
    if os.path.exists(fpath):
        return

    bf = imageio.imread(bpath)
    nuc = imageio.imread(npath)
    gt = imageio.imread(gpath)

    # Ensure all bf images have 3 channels.
    if bf.ndim == 3:
        bf = bf.transpose(2, 0, 1)
    else:
        bf = np.stack([bf] * 3, axis=0)

    # Ensure all fluo images have 3 channels.
    if nuc.ndim == 3:
        nuc = nuc.transpose(2, 0, 1)
    else:
        nuc = np.stack([nuc] * 3, axis=0)

    assert nuc.ndim == bf.ndim == 3

    # Labels have 3 channels. Keep only one.
    if gt.ndim == 3:
        gt = gt[..., 0]

    gt = connected_components(gt).astype("uint16")

    with h5py.File(fpath, "w") as f:
        f.create_dataset("raw/brightfield", data=bf, compression="gzip")
        f.create_dataset("raw/fluorescence", data=nuc, compression="gzip")
        f.create_dataset("labels", data=gt, compression="gzip")


def _preprocess_data(data_dir, base_dir):

    bf_paths = natsorted(glob(os.path.join(base_dir, "**", "brightfield", "*.png"), recursive=True))
    nucleus_paths = natsorted(glob(os.path.join(base_dir, "**", "nucleus", "*.png"), recursive=True))
    gt_paths = natsorted(glob(os.path.join(base_dir, "**", "masks", "*.png"), recursive=True))

    assert bf_paths and len(bf_paths) == len(nucleus_paths) == len(gt_paths)

    tasks = [(b, n, g, data_dir) for b, n, g in zip(bf_paths, nucleus_paths, gt_paths)]
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(_process_each_image, tasks), total=len(tasks), desc="Processing data"))


def get_aisegcell_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the aiSEGcell dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the dataset is stored.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "data.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)

    # We need to do multiple unzip and untar to get the data out.
    print(
        "'aiSEGcell' is a very large dataset (>60GB). It might take a couple of hours to download, "
        "unzip and preprocess the data. Please ensure that you have a stable internet connection."
    )
    util.unzip(zip_path=zip_path, dst=path, remove=False)
    util.unzip_tarfile(tar_path=os.path.join(path, "679085", "aisegcell_supplement.tar"), dst=path)
    util.unzip_tarfile(
        tar_path=os.path.join(path, "679085", "aiSEGcell_supplement", "data_sets", "aiSEGcell_nucleus.tar"), dst=path,
    )

    # Now that we have the core 'aiSEGcell_nucleus' folder on top-level directory, we can take it for processing data.
    _preprocess_data(data_dir=data_dir, base_dir=os.path.join(path, "aiSEGcell_nucleus"))

    return data_dir


def get_aisegcell_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False,
) -> List[str]:
    """Get paths to the aiSEGcell dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    data_dir = get_aisegcell_data(path, download)

    if split not in ["train", "val", "test"]:
        raise ValueError(f"'{split}' is not a valid split choice.")

    data_paths = glob(os.path.join(data_dir, split, "**", "*.h5"), recursive=True)
    assert len(data_paths) > 0
    return data_paths


def get_aisegcell_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    raw_channel: Literal["brightfield", "fluorescence"] = "brightfield",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the aiSEGcell dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        raw_channel: The input channel to use. Either 'brightfield' or 'fluorescence'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    data_paths = get_aisegcell_paths(path, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key=f"raw/{raw_channel}",
        label_paths=data_paths,
        label_key="labels",
        is_seg_dataset=True,
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs
    )


def get_aisegcell_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    raw_channel: Literal["brightfield", "fluorescence"] = "brightfield",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the aiSEGcell dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        raw_channel: The input channel to use. Either 'brightfield' or 'fluorescence'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_aisegcell_dataset(path, patch_shape, split, raw_channel, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
