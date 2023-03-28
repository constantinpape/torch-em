# Data loaders for the CEM datasets:
# - CEM-MitoLab: annotated data for training mitochondria segmentation models
#   - https://www.ebi.ac.uk/empiar/EMPIAR-11037/
# - CEM-1.5M: unlabeled EM images for pretraining:
#   - https://www.ebi.ac.uk/empiar/EMPIAR-11035/
# - CEM-Mito-Benchmark: 7 Benchmark datasets for mitochondria segmentation
#   - https://www.ebi.ac.uk/empiar/EMPIAR-10982/

import os
from glob import glob

import torch_em
from sklearn.model_selection import train_test_split


# TODO
def _download_cem_mitolab(path):
    # os.makedirs(path, exist_ok=True)
    raise NotImplementedError("Data download is not implemented yet for CEM data.")


def _get_cem_mitolab_paths(path, split, val_fraction, download):
    folders = glob(os.path.join(path, "*"))
    assert all(os.path.isdir(folder) for folder in folders)

    if len(folders) == 0 and download:
        _download_cem_mitolab(path)
    elif len(folders) == 0:
        raise RuntimeError(f"The CEM Mitolab data is not available at {path}, but download was set to False.")

    raw_paths, label_paths = [], []

    for folder in folders:
        images = glob(os.path.join(folder, "images", "*.tiff"))
        images.sort()
        assert len(images) > 0
        labels = glob(os.path.join(folder, "masks", "*.tiff"))
        labels.sort()
        assert len(images) == len(labels)
        raw_paths.extend(images)
        label_paths.extend(labels)

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


def get_cem_mitolab_loader(
    path, split, batch_size, patch_shape=(224, 224), val_fraction=0.05, download=False, **kwargs
):
    assert split in ("train", "val", None)
    assert os.path.exists(path)
    raw_paths, label_paths = _get_cem_mitolab_paths(path, split, val_fraction, download)
    return torch_em.default_segmentation_loader(
        raw_paths=raw_paths, raw_key=None, label_paths=label_paths, label_key=None,
        batch_size=batch_size, patch_shape=patch_shape,
        is_seg_dataset=False, ndim=2, **kwargs
    )


# TODO
def get_cem15m_loader(path):
    pass


# TODO
def get_cem_mito_benchmark_loader(path):
    pass
