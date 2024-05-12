"""Contains datasets and dataloader for the CEM data:
- CEM-MitoLab: annotated 2d data for training mitochondria segmentation models
  - https://www.ebi.ac.uk/empiar/EMPIAR-11037/
- CEM-Mito-Benchmark: 7 Benchmark datasets for mitochondria segmentation
  - https://www.ebi.ac.uk/empiar/EMPIAR-10982/
- CEM-1.5M: unlabeled EM images for pretraining: (Not yet implemented)
  - https://www.ebi.ac.uk/empiar/EMPIAR-11035/

The data itself can be downloaded from EMPIAR via aspera.
- You can install aspera via mamba. I recommend to do this in a separate environment
  to avoid dependency issues:
    - `$ mamba create -c conda-forge -c hcc -n aspera aspera-cli`
- After this you can run `$ mamba activate aspera` to have an environment with aspera installed.
- You can then download the data for one of the three datasets like this:
    - ascp -QT -l 200m -P33001 -i <PREFIX>/etc/asperaweb_id_dsa.openssh emp_ext2@fasp.ebi.ac.uk:/<EMPIAR_ID> <PATH>
    - Where <PREFIX> is the path to the mamba environment, <EMPIAR_ID> the id of one of the three datasets
      and <PATH> where you want to download the data.
- After this you can use the functions in this file if you use <PATH> as location for the data.

Note that I have implemented automatic download, but this leads to issues with
mamba for me so I recommend to download the data manually and then run the loaders
with the correct path.
"""

import json
import os
from glob import glob

import imageio.v3 as imageio
import numpy as np
import torch_em
from sklearn.model_selection import train_test_split

from .. import util

BENCHMARK_DATASETS = {
    1: "mito_benchmarks/c_elegans",
    2: "mito_benchmarks/fly_brain",
    3: "mito_benchmarks/glycolytic_muscle",
    4: "mito_benchmarks/hela_cell",
    5: "mito_benchmarks/lucchi_pp",
    6: "mito_benchmarks/salivary_gland",
    7: "tem_benchmark",
}
BENCHMARK_SHAPES = {
    1: (256, 256, 256),
    2: (256, 255, 255),
    3: (302, 383, 765),
    4: (256, 256, 256),
    5: (165, 768, 1024),
    6: (1260, 1081, 1200),
    7: (224, 224),  # NOTE: this is the minimal square shape that fits
}


def _get_mitolab_data(path, download):
    access_id = "11037"
    data_path = util.download_source_empiar(path, access_id, download)

    zip_path = os.path.join(data_path, "data/cem_mitolab.zip")
    if os.path.exists(zip_path):
        util.unzip(zip_path, data_path, remove=True)

    data_root = os.path.join(data_path, "cem_mitolab")
    assert os.path.exists(data_root)

    return data_root


def _get_all_images(path):
    raw_paths, label_paths = [], []
    folders = glob(os.path.join(path, "*"))
    assert all(os.path.isdir(folder) for folder in folders)
    for folder in folders:
        images = sorted(glob(os.path.join(folder, "images", "*.tiff")))
        assert len(images) > 0
        labels = sorted(glob(os.path.join(folder, "masks", "*.tiff")))
        assert len(images) == len(labels)
        raw_paths.extend(images)
        label_paths.extend(labels)
    return raw_paths, label_paths


def _get_non_empty_images(path):
    save_path = os.path.join(path, "non_empty_images.json")

    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            saved_images = json.load(f)
        raw_paths, label_paths = saved_images["images"], saved_images["labels"]
        raw_paths = [os.path.join(path, rp) for rp in raw_paths]
        label_paths = [os.path.join(path, lp) for lp in label_paths]
        return raw_paths, label_paths

    folders = glob(os.path.join(path, "*"))
    assert all(os.path.isdir(folder) for folder in folders)

    raw_paths, label_paths = [], []
    for folder in folders:
        images = sorted(glob(os.path.join(folder, "images", "*.tiff")))
        labels = sorted(glob(os.path.join(folder, "masks", "*.tiff")))
        assert len(images) > 0
        assert len(images) == len(labels)

        for im, lab in zip(images, labels):
            n_labels = len(np.unique(imageio.imread(lab)))
            if n_labels > 1:
                raw_paths.append(im)
                label_paths.append(lab)

    raw_paths_rel = [os.path.relpath(rp, path) for rp in raw_paths]
    label_paths_rel = [os.path.relpath(lp, path) for lp in label_paths]

    with open(save_path, "w") as f:
        json.dump({"images": raw_paths_rel, "labels": label_paths_rel}, f)

    return raw_paths, label_paths


def _get_mitolab_paths(path, split, val_fraction, download, discard_empty_images):
    data_path = _get_mitolab_data(path, download)
    if discard_empty_images:
        raw_paths, label_paths = _get_non_empty_images(data_path)
    else:
        raw_paths, label_paths = _get_all_images(data_path)

    if split is not None:
        raw_train, raw_val, labels_train, labels_val = train_test_split(
            raw_paths, label_paths, test_size=val_fraction, random_state=42,
        )
        if split == "train":
            raw_paths, label_paths = raw_train, labels_train
        else:
            raw_paths, label_paths = raw_val, labels_val

    assert len(raw_paths) > 0
    assert len(raw_paths) == len(label_paths)
    return raw_paths, label_paths


def _get_benchmark_data(path, dataset_id, download):
    access_id = "10982"
    data_path = util.download_source_empiar(path, access_id, download)
    dataset_path = os.path.join(data_path, "data", BENCHMARK_DATASETS[dataset_id])

    # these are the 3d datasets
    if dataset_id in range(1, 7):
        dataset_name = os.path.basename(dataset_path)
        raw_paths = os.path.join(dataset_path, f"{dataset_name}_em.tif")
        label_paths = os.path.join(dataset_path, f"{dataset_name}_mito.tif")
        raw_key, label_key = None, None
        is_seg_dataset = True

    # this is the 2d dataset
    else:
        raw_paths = os.path.join(dataset_path, "images")
        label_paths = os.path.join(dataset_path, "masks")
        raw_key, label_key = "*.tiff", "*.tiff"
        is_seg_dataset = False

    return raw_paths, label_paths, raw_key, label_key, is_seg_dataset


#
# data sets
#


def get_mitolab_dataset(
    path, split, patch_shape=(224, 224), val_fraction=0.05, download=False,
    discard_empty_images=True, **kwargs
):
    assert split in ("train", "val", None)
    assert os.path.exists(path)
    raw_paths, label_paths = _get_mitolab_paths(path, split, val_fraction, download, discard_empty_images)
    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths, raw_key=None,
        label_paths=label_paths, label_key=None,
        patch_shape=patch_shape, is_seg_dataset=False, ndim=2, **kwargs
    )


def get_cem15m_dataset(path):
    raise NotImplementedError


def get_benchmark_dataset(
    path, dataset_id, patch_shape, download=False, **kwargs,
):
    """
    ascp -QT -l 200m -P33001 -i <CONDA_PREFIX>/etc/asperaweb_id_dsa.openssh emp_ext2@fasp.ebi.ac.uk:/10982 <PATH>
    """
    if dataset_id not in range(1, 8):
        raise ValueError
    raw_paths, label_paths, raw_key, label_key, is_seg_dataset = _get_benchmark_data(path, dataset_id, download)
    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths, raw_key=raw_key,
        label_paths=label_paths, label_key=label_key,
        patch_shape=patch_shape,
        is_seg_dataset=is_seg_dataset, **kwargs,
    )


#
# data loaders
#


def get_mitolab_loader(
    path, split, batch_size, patch_shape=(224, 224),
    discard_empty_images=True,
    val_fraction=0.05, download=False, **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_mitolab_dataset(
        path, split, patch_shape, download=download, discard_empty_images=discard_empty_images, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader


def get_cem15m_loader(path):
    raise NotImplementedError


def get_benchmark_loader(path, dataset_id, batch_size, patch_shape, download=False, **kwargs):
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_benchmark_dataset(
        path, dataset_id,
        patch_shape=patch_shape, download=download, **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
