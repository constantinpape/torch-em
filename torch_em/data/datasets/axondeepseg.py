import os
from glob import glob
from shutil import rmtree

import imageio
import h5py
import numpy as np
import torch_em

from . import util

URLS = {
    "sem": "https://github.com/axondeepseg/data_axondeepseg_sem/archive/refs/heads/master.zip",
    "tem": "https://osf.io/download/uewd9"
}
CHECKSUMS = {
    "sem": "d334cbacf548f78ce8dd4a597bf86b884bd15a47a230a0ccc46e1ffa94d58426",
    "tem": "e4657280808f3b80d3bf1fba87d1cbbf2455f519baf1a7b16d2ddf2e54739a95"
}


def _preprocess_sem_data(out_path):
    # preprocess the data to get it to a better data format
    data_root = os.path.join(out_path, "data_axondeepseg_sem-master")
    assert os.path.exists(data_root)

    # get the raw data paths
    raw_folders = glob(os.path.join(data_root, "sub-rat*"))
    raw_folders.sort()
    raw_paths = []
    for folder in raw_folders:
        paths = glob(os.path.join(folder, "micr", "*.png"))
        paths.sort()
        raw_paths.extend(paths)

    # get the label paths
    label_folders = glob(os.path.join(
        data_root, "derivatives", "labels", "sub-rat*"
    ))
    label_folders.sort()
    label_paths = []
    for folder in label_folders:
        paths = glob(os.path.join(folder, "micr", "*axonmyelin-manual.png"))
        paths.sort()
        label_paths.extend(paths)
    assert len(raw_paths) == len(label_paths), f"{len(raw_paths)}, {len(label_paths)}"

    # process raw data and labels
    for i, (rp, lp) in enumerate(zip(raw_paths, label_paths)):
        outp = os.path.join(out_path, f"sem_data_{i}.h5")
        with h5py.File(outp, "w") as f:

            # raw data: invert to match tem em intensities
            raw = imageio.imread(rp)
            assert np.dtype(raw.dtype) == np.dtype("uint8")
            if raw.ndim == 3:  # (one of the images is RGBA)
                raw = np.mean(raw[..., :-3], axis=-1)
            raw = 255 - raw
            f.create_dataset("raw", data=raw, compression="gzip")

            # labels: map from
            # 0 -> 0
            # 127, 128 -> 1
            # 255 -> 2
            labels = imageio.imread(lp)
            assert labels.shape == raw.shape, f"{labels.shape}, {raw.shape}"
            label_vals = np.unique(labels)
            # 127, 128: both myelin labels, 130, 233: noise
            assert len(np.setdiff1d(label_vals, [0, 127, 128, 130, 233, 255])) == 0, f"{label_vals}"
            new_labels = np.zeros_like(labels)
            new_labels[labels == 127] = 1
            new_labels[labels == 128] = 1
            new_labels[labels == 255] = 2
            f.create_dataset("labels", data=new_labels, compression="gzip")

    # clean up
    rmtree(data_root)


def _preprocess_tem_data(out_path):
    data_root = os.path.join(out_path, "TEM_dataset")
    folder_names = os.listdir(data_root)
    folders = [os.path.join(data_root, fname) for fname in folder_names
               if os.path.isdir(os.path.join(data_root, fname))]
    for i, folder in enumerate(folders):
        data_out = os.path.join(out_path, f"tem_{i}.h5")
        with h5py.File(data_out, "w") as f:
            im = imageio.imread(os.path.join(folder, "image.png"))
            f.create_dataset("raw", data=im, compression="gzip")

            # labels: map from
            # 0 -> 0
            # 128 -> 1
            # 255 -> 2
            # the rest are noise
            labels = imageio.imread(os.path.join(folder, "mask.png"))
            new_labels = np.zeros_like(labels)
            new_labels[labels == 128] = 1
            new_labels[labels == 255] = 2
            f.create_dataset("labels", data=new_labels, compression="gzip")

    # clean up
    rmtree(data_root)


def _require_axondeepseg_data(path, name, download):

    # download and unzip the data
    url, checksum = URLS[name], CHECKSUMS[name]
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, name)
    if os.path.exists(out_path):
        return out_path

    tmp_path = os.path.join(path, f"{name}.zip")
    util.download_source(tmp_path, url, download, checksum=checksum)
    util.unzip(tmp_path, out_path, remove=True)

    if name == "sem":
        _preprocess_sem_data(out_path)
    elif name == "tem":
        _preprocess_tem_data(out_path)

    return out_path


def get_axondeepseg_dataset(
    path, name, patch_shape, download=False, one_hot_encoding=False, val_fraction=None, split=None, **kwargs
):
    """Dataset for the segmentation of myelinated axons in EM.

    This dataset is from the publication https://doi.org/10.1038/s41598-018-22181-4.
    Please cite it if you use this dataset for a publication.
    """
    if isinstance(name, str):
        name = [name]
    assert isinstance(name, (tuple, list))

    all_paths = []
    for nn in name:
        data_root = _require_axondeepseg_data(path, nn, download)
        paths = glob(os.path.join(data_root, "*.h5"))
        paths.sort()
        if val_fraction is not None:
            assert split is not None
            n_samples = int(len(paths) * (1 - val_fraction))
            paths = paths[:n_samples] if split == "train" else paths[n_samples:]
        all_paths.extend(paths)

    if one_hot_encoding:
        if isinstance(one_hot_encoding, bool):
            # add transformation to go from [0, 1, 2] to one hot encoding
            class_ids = [0, 1, 2]
        elif isinstance(one_hot_encoding, int):
            class_ids = list(range(one_hot_encoding))
        elif isinstance(one_hot_encoding, (list, tuple)):
            class_ids = list(one_hot_encoding)
        else:
            raise ValueError(
                f"Invalid value {one_hot_encoding} passed for 'one_hot_encoding', expect bool, int or list."
            )
        label_transform = torch_em.transform.label.OneHotTransform(class_ids=class_ids)
        msg = "'one_hot' is set to True, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = util.update_kwargs(kwargs, "label_transform", label_transform, msg=msg)

    raw_key, label_key = "raw", "labels"
    return torch_em.default_segmentation_dataset(all_paths, raw_key, all_paths, label_key, patch_shape, **kwargs)


# add instance segmentation representations?
def get_axondeepseg_loader(
    path, name, patch_shape, batch_size,
    download=False, one_hot_encoding=False,
    val_fraction=None, split=None, **kwargs
):
    """Dataloader for the segmentation of myelinated axons. See 'get_axondeepseg_dataset' for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_axondeepseg_dataset(
        path, name, patch_shape, download=download, one_hot_encoding=one_hot_encoding,
        val_fraction=val_fraction, split=split, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
