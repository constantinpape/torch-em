import os
import multiprocessing
from concurrent import futures
from shutil import rmtree

import imageio
import numpy as np
import torch_em
import z5py

from tqdm import tqdm
from . import util

# TODO: update the links to the new host location at huggingface.
# - https://mitoem.grand-challenge.org/ (see `Dataset` for the links)

URLS = {
    "raw": {
        "human": "https://www.dropbox.com/s/z41qtu4y735j95e/EM30-H-im.zip?dl=1",
        "rat": "https://huggingface.co/datasets/pytc/EM30/resolve/main/EM30-R-im.zip"
    },
    "labels": {
        "human": "https://www.dropbox.com/s/dhf89bc14kemw4e/EM30-H-mito-train-val-v2.zip?dl=1",
        "rat": "https://www.dropbox.com/s/stncdytayhr8ggz/EM30-R-mito-train-val-v2.zip?dl=1"
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


def get_mitoem_dataset(
    path,
    splits,
    patch_shape,
    samples=("human", "rat"),
    download=False,
    offsets=None,
    boundaries=False,
    binary=False,
    **kwargs,
):
    """Dataset for the segmentation of mitochondria in EM.

    This dataset is from the publication https://doi.org/10.1007/978-3-030-59722-1_7.
    Please cite it if you use this dataset for a publication.
    """
    assert len(patch_shape) == 3
    if isinstance(splits, str):
        splits = [splits]
    assert len(set(splits) - {"train", "val"}) == 0, f"{splits}"
    assert len(set(samples) - {"human", "rat"}) == 0, f"{samples}"
    os.makedirs(path, exist_ok=True)

    data_paths = []
    for sample in samples:
        if not _check_data(path, sample):
            print("The MitoEM data for sample", sample, "is not available yet and will be downloaded and created.")
            print("Note that this dataset is large, so this step can take several hours (depending on your internet).")
            _require_mitoem_sample(path, sample, download)
            print("The MitoEM data for sample", sample, "has been created.")

        for split in splits:
            split_path = os.path.join(path, f"{sample}_{split}.n5")
            assert os.path.exists(split_path), split_path
            data_paths.append(split_path)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )
    raw_key = "raw"
    label_key = "labels"
    return torch_em.default_segmentation_dataset(data_paths, raw_key, data_paths, label_key, patch_shape, **kwargs)


def get_mitoem_loader(
    path,
    splits,
    patch_shape,
    batch_size,
    samples=("human", "rat"),
    download=False,
    offsets=None,
    boundaries=False,
    binary=False,
    **kwargs,
):
    """Dataloader for the segmentation of mitochondria in EM. See 'get_mitoem_dataset' for details."""
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_mitoem_dataset(
        path, splits, patch_shape,
        samples=samples, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
