import os
from concurrent import futures
from glob import glob
from shutil import rmtree

import imageio
import h5py
import numpy as np
import torch_em

from tqdm import tqdm
from .. import util

URL = "http://www.casser.io/files/kasthuri_pp.zip "
CHECKSUM = "bbb78fd205ec9b57feb8f93ebbdf1666261cbc3e0305e7f11583ab5157a3d792"

# data from: https://sites.google.com/view/connectomics/
# TODO: add sampler for foreground (-1 is empty area)
# TODO and masking for the empty space


def _load_volume(path):
    files = glob(os.path.join(path, "*.png"))
    files.sort()
    nz = len(files)

    im0 = imageio.imread(files[0])
    out = np.zeros((nz,) + im0.shape, dtype=im0.dtype)
    out[0] = im0

    def _loadz(z):
        im = imageio.imread(files[z])
        out[z] = im

    n_threads = 8
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(_loadz, range(1, nz)), desc="Load volume", total=nz-1
        ))

    return out


def _create_data(root, inputs, out_path):
    raw = _load_volume(os.path.join(root, inputs[0]))
    labels_argb = _load_volume(os.path.join(root, inputs[1]))
    assert labels_argb.ndim == 4
    labels = np.zeros(raw.shape, dtype="int8")

    fg_mask = (labels_argb == np.array([255, 255, 255])[None, None, None]).all(axis=-1)
    labels[fg_mask] = 1
    bg_mask = (labels_argb == np.array([2, 2, 2])[None, None, None]).all(axis=-1)
    labels[bg_mask] = -1
    assert (np.unique(labels) == np.array([-1, 0, 1])).all()
    assert raw.shape == labels.shape, f"{raw.shape}, {labels.shape}"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")


def get_kasthuri_data(path, download):
    """Download the kasthuri dataset. See `get_kasthuri_dataset` for details.
    """
    if os.path.exists(path):
        return path

    os.makedirs(path)
    tmp_path = os.path.join(path, "kasthuri.zip")
    util.download_source(tmp_path, URL, download, checksum=CHECKSUM)
    util.unzip(tmp_path, path, remove=True)

    root = os.path.join(path, "Kasthuri++")
    assert os.path.exists(root), root

    inputs = [["Test_In", "Test_Out"], ["Train_In", "Train_Out"]]
    outputs = ["kasthuri_train.h5", "kasthuri_test.h5"]
    for inp, out in zip(inputs, outputs):
        out_path = os.path.join(path, out)
        _create_data(root, inp, out_path)

    rmtree(root)


def get_kasthuri_dataset(path, split, patch_shape, download=False, **kwargs):
    """Dataset for the segmentation of mitochondria in EM.

    This dataset is from the publication https://doi.org/10.48550/arXiv.1812.06024.
    Please cite it if you use this dataset for a publication.
    """
    assert split in ("train", "test")
    get_kasthuri_data(path, download)
    data_path = os.path.join(path, f"kasthuri_{split}.h5")
    assert os.path.exists(data_path), data_path
    raw_key, label_key = "raw", "labels"
    return torch_em.default_segmentation_dataset(data_path, raw_key, data_path, label_key, patch_shape, **kwargs)


def get_kasthuri_loader(path, split, patch_shape, batch_size, download=False, **kwargs):
    """Dataloader for the segmentation of mitochondria in EM. See 'get_kasthuri_dataset' for details."""
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_kasthuri_dataset(path, split, patch_shape, download=download, **ds_kwargs)
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
