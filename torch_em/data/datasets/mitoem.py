import os
import multiprocessing
from shutil import rmtree

import imageio
import numpy as np
import torch_em
import z5py
from tqdm import tqdm
from .util import download_source, update_kwargs, unzip

URLS = {
    "human": "https://www.dropbox.com/sh/p5xn9e4gderjtm6/AADfUMzAA38XBvcXDTG1kAGGa/MitoEM-H.zip?dl=1",
    "rat": "https://www.dropbox.com/sh/p5xn9e4gderjtm6/AAAjVuVydTvccP1D4SvakrLda/MitoEM-R.zip?dl=1"
}
CHECKSUMS = {
    "human": "f4ad14e098697be78d3ea13f263f76d5ba81a27e354c9edc906adfe728c765bd",
    "rat": "6da3ad29ae867eb13819a28fe04e8d771489b07e4ff5a854f8a369b72abc346f"
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


def _load_vol(pattern, slice_ids, desc, dtype=None):
    im0 = pattern % slice_ids[0]
    im0 = imageio.imread(im0)

    shape = (len(slice_ids),) + im0.shape

    dtype = im0.dtype if dtype is None else dtype
    out = np.zeros(shape, dtype=dtype)
    out[0] = im0

    for z, slice_id in tqdm(enumerate(slice_ids[1:], 1), total=len(slice_ids) - 1, desc=desc):
        out[z] = imageio.imread(pattern % slice_id)

    return out


def _create_volume(out_path, im_folder, label_folder=None, z_start=None):
    if label_folder is None:
        assert z_start is not None
        n_slices = len(get_slices(im_folder))
        slices = list(range(z_start, n_slices))
    else:
        assert z_start is None
        slices = get_slices(label_folder)

    raw = _load_vol(os.path.join(im_folder, "im%04i.png"), slices, "load raw")
    if label_folder is not None:
        labels = _load_vol(os.path.join(label_folder, "seg%04i.tif"), slices, "load labels", "uint64")

    print("Write volume to", out_path)
    chunks = (32, 256, 256)
    n_threads = min(16, multiprocessing.cpu_count())
    with z5py.File(out_path, "a") as f:
        f.create_dataset("raw", data=raw, chunks=chunks, compression="gzip", n_threads=n_threads)
        if label_folder is not None:
            ds = f.create_dataset("labels", data=labels, chunks=chunks, compression="gzip", n_threads=n_threads)
            ds.attrs["maxId"] = int(labels.max()) + 1

    return slices[-1]


def _require_mitoem_sample(path, sample, download):
    url = URLS[sample]
    checksum = CHECKSUMS[sample]
    zip_path = os.path.join(path, f"{sample}.zip")
    download_source(zip_path, url, download, checksum)
    unzip(zip_path, path, remove=True)

    prefix = "MitoEM-H" if sample == "human" else "MitoEM-R"
    im_folder = os.path.join(path, prefix, "im")
    train_folder = os.path.join(path, prefix, "mito_train")
    val_folder = os.path.join(path, prefix, "mito_val")

    print("Create train volume")
    train_path = os.path.join(path, f"{sample}_train.n5")
    _create_volume(train_path, im_folder, train_folder)

    print("Create validation volume")
    val_path = os.path.join(path, f"{sample}_val.n5")
    z = _create_volume(val_path, im_folder, val_folder)

    print("Create test volume")
    test_path = os.path.join(path, f"{sample}_test.n5")
    _create_volume(test_path, im_folder, z_start=z)

    rmtree(os.path.join(path, prefix))


def get_mitoem_loader(
    path,
    patch_shape,
    splits,
    samples=("human", "rat"),
    download=False,
    offsets=None,
    boundaries=False,
    binary=False,
    **kwargs,
):
    """
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

    assert sum((offsets is not None, boundaries, binary)) <= 1, f"{offsets}, {boundaries}, {binary}"
    if offsets is not None:
        # we add a binary target channel for foreground background segmentation
        label_transform = torch_em.transform.label.AffinityTransform(offsets=offsets,
                                                                     ignore_label=None,
                                                                     add_binary_target=True,
                                                                     add_mask=True)
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, 'label_transform2', label_transform, msg=msg)
    elif boundaries:
        label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)
        msg = "Boundaries is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, 'label_transform', label_transform, msg=msg)
    elif binary:
        label_transform = torch_em.transform.label.labels_to_binary
        msg = "Binary is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, 'label_transform', label_transform, msg=msg)

    raw_key = "raw"
    label_key = "labels"
    kwargs["ndim"] = 3
    return torch_em.default_segmentation_loader(data_paths, raw_key, data_paths, label_key,
                                                patch_shape=patch_shape, **kwargs)
