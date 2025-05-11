"""
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import List, Union, Tuple, Literal

import imageio.v3 as imageio
from skimage.measure import label as connected_components

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://libdrive.ethz.ch/index.php/s/VoF2SYkbLY8izjh/download"
CHECKSUM = "f9115ee6b71e7c4364b83f7d7f8b66dce5b778344070bddb6a8f0e5086ca5de9"


def _preprocess_data(data_dir, base_dir):
    import h5py

    bf_paths = natsorted(glob(os.path.join(base_dir, "**", "brightfield", "*.png"), recursive=True))
    nucleus_paths = natsorted(glob(os.path.join(base_dir, "**", "nucleus", "*.png"), recursive=True))
    gt_paths = natsorted(glob(os.path.join(base_dir, "**", "masks", "*.png"), recursive=True))

    assert bf_paths and len(bf_paths) == len(nucleus_paths) == len(gt_paths)

    for bpath, npath, gpath in tqdm(
        zip(bf_paths, nucleus_paths, gt_paths), desc="Processing data", total=len(bf_paths)
    ):
        bf = imageio.imread(bpath)
        nuc = imageio.imread(npath)
        gt = imageio.imread(gpath)
        gt = connected_components(gt).astype("uint16")

        # Let's get the split info first.
        path_parents = Path(bpath).parents
        split = path_parents[1].name.split("_")[-1]
        dname = path_parents[2].name

        neu_dir = os.path.join(data_dir, split, dname)
        os.makedirs(neu_dir, exist_ok=True)

        fpath = os.path.join(neu_dir, f"{Path(bpath).stem}.h5")

        with h5py.File(fpath, "w") as f:
            f.create_dataset("raw/brightfield", data=bf, compression="gzip")
            f.create_dataset("raw/fluorescence", data=nuc, compression="gzip")
            f.create_dataset("labels", data=gt, compression="gzip")


def get_aisegcell_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    # zip_path = os.path.join(path, "data.zip")
    # util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)

    # # We need to do multiple unzip and untar to get the data out.
    print(
        "'aiSEGcell' is a very large dataset (>60GB). It might take a couple of hours to download, "
        "unzip and preprocess the data. Please ensure that you have a stable internet connection."
    )
    # util.unzip(zip_path=zip_path, dst=path, remove=False)
    # util.unzip_tarfile(tar_path=os.path.join(path, "679085", "aisegcell_supplement.tar"), dst=path)
    util.unzip_tarfile(
        tar_path=os.path.join(path, "679085", "aiSEGcell_supplement", "data_sets", "aiSEGcell_nucleus.tar"), dst=path,
    )

    # Now that we have the core 'aiSEGcell_nucleus' folder on top-level directory, we can take it for processing data.
    _preprocess_data(data_dir=data_dir, base_dir=os.path.join(path, "aiSEGcell_nucleus"))

    return data_dir


def get_aisegcell_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False,
) -> List[str]:
    """
    """
    data_dir = get_aisegcell_data(path, download)
    data_paths = glob(os.path.join(data_dir, split, "*.h5"))
    return data_paths


def get_aisegcell_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    raw_channel: Literal["brightfield", "fluorescence"] = "brightfield",
    download: bool = False,
    **kwargs
) -> Dataset:
    """
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
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_aisegcell_dataset(path, patch_shape, split, raw_channel, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
