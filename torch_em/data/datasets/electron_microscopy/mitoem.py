"""MitoEM is a dataset for segmenting mitochondria in electron microscopy.
It contains two large annotated volumes, one from rat cortex, the other from human cortex.
This dataset was used for a segmentation challenge at ISBI 2022.

If you use it in your research then please cite https://doi.org/10.1007/978-3-030-59722-1_7.
"""

import os
from tqdm import tqdm
import multiprocessing
from shutil import rmtree
from concurrent import futures
from typing import List, Optional, Sequence, Tuple, Union

import imageio
import numpy as np

import torch_em

from torch.utils.data import Dataset, DataLoader

from .. import util


URLS = {
    "raw": {
        "human": "https://www.dropbox.com/s/z41qtu4y735j95e/EM30-H-im.zip?dl=1",
        "rat": "https://huggingface.co/datasets/pytc/EM30/resolve/main/EM30-R-im.zip"
    },
    "labels": {
        "human": "https://www.dropbox.com/s/dhf89bc14kemw4e/EM30-H-mito-train-val-v2.zip?dl=1",
        "rat": "https://huggingface.co/datasets/pytc/MitoEM/resolve/main/EM30-R-mito-train-val-v2.zip"
    }
}
CHECKSUMS = {
    "raw": {
        "human": "98fe259f36a7d8d43f99981b7a0ef8cdeba2ce2615ff91595f428ae57207a041",
        "rat": "6a2cac68adde5d01984542d3ee1d7753d1fa3e6eb2a042ce15ce297c95885bbe"
    },
    "labels": {
        "human": "0e8ed292cfcd0c58701d9f4299244a1b66d6aeb506c85754c34f98a4eda0ef1b",
        "rat": "c56380ac575428a818bd293ca3509d1249999846c3702ccbf11d308acdd2ae86"
    }
}


def _check_data(path, sample):
    splits = ["train", "val", "test"]
    expected_paths = [os.path.join(path, f"{sample}_{split}.n5") for split in splits]
    return all(os.path.exists(pp) for pp in expected_paths)


def get_slices(folder):
    files = os.listdir(folder)
    files.sort()
    files = [os.path.splitext(ff)[0] for ff in files]
    slice_ids = [int(ff[2:]) if ff.startswith('im') else int(ff[3:]) for ff in files]
    return slice_ids


def _load_vol(pattern, slice_ids, desc, n_threads, dtype=None):
    im0 = pattern % slice_ids[0]
    im0 = imageio.imread(im0)

    shape = (len(slice_ids),) + im0.shape

    dtype = im0.dtype if dtype is None else dtype
    out = np.zeros(shape, dtype=dtype)
    out[0] = im0

    def load_slice(z, slice_id):
        out[z] = imageio.imread(pattern % slice_id)

    zs = list(range(1, len(slice_ids)))
    assert len(zs) == len(slice_ids) - 1
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(load_slice, zs, slice_ids[1:]), total=len(slice_ids) - 1, desc=desc))

    return out


def _create_volume(out_path, im_folder, label_folder=None, z_start=None):
    import z5py

    if label_folder is None:
        assert z_start is not None
        n_slices = len(get_slices(im_folder))
        slices = list(range(z_start, n_slices))
    else:
        assert z_start is None
        slices = get_slices(label_folder)

    n_threads = min(16, multiprocessing.cpu_count())
    raw = _load_vol(os.path.join(im_folder, "im%04i.png"), slices, "load raw", n_threads)
    if label_folder is not None:
        labels = _load_vol(os.path.join(label_folder, "seg%04i.tif"), slices, "load labels", n_threads, dtype="uint64")

    print("Write volume to", out_path)
    chunks = (32, 256, 256)
    with z5py.File(out_path, "a") as f:
        f.create_dataset("raw", data=raw, chunks=chunks, compression="gzip", n_threads=n_threads)
        if label_folder is not None:
            ds = f.create_dataset("labels", data=labels, chunks=chunks, compression="gzip", n_threads=n_threads)
            ds.attrs["maxId"] = int(labels.max()) + 1

    return slices[-1]


def _require_mitoem_sample(path, sample, download):
    os.makedirs(path, exist_ok=True)

    for name in ("raw", "labels"):
        url = URLS[name][sample]
        checksum = CHECKSUMS[name][sample]
        zip_path = os.path.join(path, f"{sample}.zip")
        util.download_source(zip_path, url, download, checksum)
        util.unzip(zip_path, path, remove=True)

    im_folder = os.path.join(path, "im")
    train_folder = os.path.join(path, "mito-train-v2")
    val_folder = os.path.join(path, "mito-val-v2")

    print("Create train volume")
    train_path = os.path.join(path, f"{sample}_train.n5")
    _create_volume(train_path, im_folder, train_folder)

    print("Create validation volume")
    val_path = os.path.join(path, f"{sample}_val.n5")
    z = _create_volume(val_path, im_folder, val_folder)

    print("Create test volume")
    test_path = os.path.join(path, f"{sample}_test.n5")
    _create_volume(test_path, im_folder, z_start=z)

    rmtree(im_folder)
    rmtree(train_folder)
    rmtree(val_folder)


def get_mitoem_data(path: Union[os.PathLike, str], samples: Sequence[str], splits: Sequence[str], download: bool):
    """Download the MitoEM training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        samples: The samples to download. The available samples are 'human' and 'rat'.
        splits: The data splits to download. The available splits are 'train', 'val' and 'test'.
        download: Whether to download the data if it is not present.
    """
    if isinstance(splits, str):
        splits = [splits]
    assert len(set(splits) - {"train", "val"}) == 0, f"{splits}"
    assert len(set(samples) - {"human", "rat"}) == 0, f"{samples}"
    os.makedirs(path, exist_ok=True)

    for sample in samples:
        if not _check_data(path, sample):
            print("The MitoEM data for sample", sample, "is not available yet and will be downloaded and created.")
            print("Note that this dataset is large, so this step can take several hours (depending on your internet).")
            _require_mitoem_sample(path, sample, download)
            print("The MitoEM data for sample", sample, "has been created.")

        for split in splits:
            split_path = os.path.join(path, f"{sample}_{split}.n5")
            assert os.path.exists(split_path), split_path


def get_mitoem_paths(
    path: Union[os.PathLike, str],
    splits: Sequence[str],
    samples: Sequence[str] = ("human", "rat"),
    download: bool = False,
) -> List[str]:
    """Get paths for MitoEM data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        samples: The samples to download. The available samples are 'human' and 'rat'.
        splits: The data splits to download. The available splits are 'train', 'val' and 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepaths for the stored data.
    """
    get_mitoem_data(path, samples, splits, download)
    data_paths = [os.path.join(path, f"{sample}_{split}.n5") for split in splits for sample in samples]
    return data_paths


def get_mitoem_dataset(
    path: Union[os.PathLike, str],
    splits: Sequence[str],
    patch_shape: Tuple[int, int, int],
    samples: Sequence[str] = ("human", "rat"),
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs,
) -> Dataset:
    """Get the MitoEM dataset for the segmentation of mitochondria in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        splits: The splits to use for the dataset. Available values are 'train', 'val' and 'test'.
        patch_shape: The patch shape to use for training.
        samples: The samples to use for the dataset. The available samples are 'human' and 'rat'.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    assert len(patch_shape) == 3

    data_paths = get_mitoem_paths(path, samples, splits, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="raw",
        label_paths=data_paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs
    )


def get_mitoem_loader(
    path: Union[os.PathLike, str],
    splits: Sequence[str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    samples: Sequence[str] = ("human", "rat"),
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the MitoEM dataloader for the segmentation of mitochondria in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        splits: The splits to use for the dataset. Available values are 'train', 'val' and 'test'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        samples: The samples to use for the dataset. The available samples are 'human' and 'rat'.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to return a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
       The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_mitoem_dataset(
        path, splits, patch_shape, samples=samples, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
