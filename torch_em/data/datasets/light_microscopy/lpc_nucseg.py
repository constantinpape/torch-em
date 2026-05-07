"""The LPC NucSeg dataset contains annotations for nuclear segmentation
in fluorescence microscopy images.

The dataset provides 97 hand-segmented images with ~4,009 cells from U2OS (gnf)
and NIH3T3 (ic100) cell lines.

The dataset is located at https://github.com/luispedro/Coelho2009_ISBI_NuclearSegmentation.
This dataset is from the publication https://doi.org/10.1109/ISBI.2009.5193098.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Union, Tuple, List, Optional

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "images": "https://github.com/luispedro/Coelho2009_ISBI_NuclearSegmentation/archive/refs/heads/master.zip",
}


def _create_h5_data(path, source):
    """Create h5 files with raw images and instance labels."""
    import h5py
    from tqdm import tqdm

    repo_dir = os.path.join(path, "Coelho2009_ISBI_NuclearSegmentation-master")
    h5_dir = os.path.join(path, "h5_data", source)
    os.makedirs(h5_dir, exist_ok=True)

    raw_dir = os.path.join(repo_dir, "data", "images", "dna-images", source)
    label_dir = os.path.join(repo_dir, "data", "preprocessed-data", source)

    raw_paths = sorted(glob(os.path.join(raw_dir, "*.png")))

    for raw_path in tqdm(raw_paths, desc=f"Creating h5 files for {source}"):
        fname = os.path.basename(raw_path)
        h5_path = os.path.join(h5_dir, fname.replace(".png", ".h5"))

        if os.path.exists(h5_path):
            continue

        label_path = os.path.join(label_dir, fname)
        if not os.path.exists(label_path):
            continue

        raw = imageio.imread(raw_path)
        labels = imageio.imread(label_path)

        # Convert RGB to grayscale if needed (DNA fluorescence should be single channel)
        if raw.ndim == 3:
            raw = raw[..., 0]  # Take first channel

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels.astype("int64"), compression="gzip")

    return h5_dir


def get_lpc_nucseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the LPC NucSeg dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the directory with the data.
    """
    repo_dir = os.path.join(path, "Coelho2009_ISBI_NuclearSegmentation-master")
    if os.path.exists(repo_dir):
        return repo_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "master.zip")
    util.download_source(path=zip_path, url=URLS["images"], download=download, checksum=None)
    util.unzip(zip_path=zip_path, dst=path, remove=False)

    return repo_dir


def get_lpc_nucseg_paths(
    path: Union[os.PathLike, str],
    source: Optional[Union[str, List[str]]] = None,
    download: bool = False,
) -> List[str]:
    """Get paths to the LPC NucSeg data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        source: The data source(s) to use. One of 'gnf' (U2OS cells) or 'ic100' (NIH3T3 cells).
            Can also be a list of sources. If None, all sources will be used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the h5 data.
    """
    from natsort import natsorted

    get_lpc_nucseg_data(path, download)

    if source is None:
        source = ["gnf", "ic100"]
    elif isinstance(source, str):
        source = [source]

    all_h5_paths = []
    for src in source:
        assert src in ("gnf", "ic100"), f"'{src}' is not a valid source. Choose from 'gnf' or 'ic100'."

        h5_dir = os.path.join(path, "h5_data", src)
        if not os.path.exists(h5_dir) or len(glob(os.path.join(h5_dir, "*.h5"))) == 0:
            _create_h5_data(path, src)

        h5_paths = glob(os.path.join(h5_dir, "*.h5"))
        all_h5_paths.extend(h5_paths)

    assert len(all_h5_paths) > 0, f"No data found for source '{source}'"

    return natsorted(all_h5_paths)


def get_lpc_nucseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    source: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LPC NucSeg dataset for nuclear segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        source: The data source(s) to use. One of 'gnf' (U2OS cells) or 'ic100' (NIH3T3 cells).
            Can also be a list of sources. If None, all sources will be used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    h5_paths = get_lpc_nucseg_paths(path, source, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True,
    )
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=h5_paths,
        raw_key="raw",
        label_paths=h5_paths,
        label_key="labels",
        patch_shape=patch_shape,
        ndim=2,
        **kwargs
    )


def get_lpc_nucseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    source: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LPC NucSeg dataloader for nuclear segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        source: The data source(s) to use. One of 'gnf' (U2OS cells) or 'ic100' (NIH3T3 cells).
            Can also be a list of sources. If None, all sources will be used.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lpc_nucseg_dataset(
        path=path,
        patch_shape=patch_shape,
        source=source,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
