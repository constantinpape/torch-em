"""This dataset contains annotations for nucleus instance segmentation and classification
in H&E stained colon histopathology images, extending the Lizard dataset with an additional
mitosis class. It provides two subsets: 'lizard', a modified version of the Lizard dataset
with mitosis annotations added, and 'mitosis', a dedicated mitosis train / validation / test
split (both originally released together as the 'lizard_mitosis' and 'mitosis_ds' resources).

This dataset is from the publication https://proceedings.mlr.press/v250/baumann24a.html.
Please cite it if you use this dataset for your research.
"""

import os
from tqdm import tqdm
from typing import Tuple, Union, Literal

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "lizard": "https://zenodo.org/records/10636591/files/lizard_mitosis.zip",
    "mitosis": "https://zenodo.org/records/10636591/files/mitosis_ds.zip",
}
CHECKSUMS = {
    "lizard": "5859d738891f4620914a3fab317b3800fde8a5cd5cb4edf8003fd0efdc255ab3",
    "mitosis": "4b2b5d49e52611d5937f94e6cc06d8c642a0323e2250693ef465146382bde736",
}
SPLIT_FILES = {
    "lizard": {
        "train": ("fold_0/train_img.npy", "fold_0/train_lab.npy"),
        "val": ("fold_0/valid_img.npy", "fold_0/valid_lab.npy"),
        "test": ("test_images.npy", "test_labels.npy"),
    },
    "mitosis": {
        "train": ("train_full_img.npy", "train_full_lab.npy"),
        "val": ("valid_full_img.npy", "valid_full_lab.npy"),
        "test": ("test_ds/test_img.npy", "test_ds/test_lab.npy"),
    },
}


def _extract_split(data_dir, subset, split):
    import h5py

    out_path = os.path.join(data_dir, f"{split}.h5")
    if os.path.exists(out_path):
        return out_path

    img_file, lab_file = SPLIT_FILES[subset][split]
    images = np.load(os.path.join(data_dir, img_file), mmap_mode="r")
    labels = np.load(os.path.join(data_dir, lab_file), mmap_mode="r")
    assert images.shape[0] == labels.shape[0], (images.shape, labels.shape)

    n_samples = images.shape[0]
    tmp_path = f"{out_path}.incomplete"
    with h5py.File(tmp_path, "a") as f:
        raw = f.create_dataset("raw", shape=(3, n_samples) + images.shape[1:3], dtype=images.dtype)
        instances = f.create_dataset("labels/instances", shape=(n_samples,) + images.shape[1:3], dtype=labels.dtype)
        semantic = f.create_dataset("labels/semantic", shape=(n_samples,) + images.shape[1:3], dtype=labels.dtype)
        for i in tqdm(range(n_samples), desc=f"Extract {subset} '{split}' data"):
            raw[:, i] = images[i].transpose(2, 0, 1)
            instances[i] = labels[i, ..., 0]
            semantic[i] = labels[i, ..., 1]

    os.replace(tmp_path, out_path)
    return out_path


def get_lizard_mitosis_data(
    path: Union[os.PathLike, str], subset: Literal["lizard", "mitosis"], download: bool = False
) -> str:
    """Download the lizard-mitosis dataset for nucleus segmentation and classification.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        subset: The choice of data subset. Either 'lizard' (modified Lizard dataset with an
            added mitosis class) or 'mitosis' (dedicated mitosis dataset).
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data directory.
    """
    if subset not in URLS:
        raise ValueError(f"'{subset}' is not a valid subset.")

    data_dir = os.path.join(path, subset)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"{subset}.zip")
    util.download_source(zip_path, URLS[subset], download, checksum=CHECKSUMS[subset])
    util.unzip(zip_path, path)

    extracted_dir = os.path.join(path, "lizard_mitosis" if subset == "lizard" else "mitosis_ds")
    os.rename(extracted_dir, data_dir)

    return data_dir


def get_lizard_mitosis_paths(
    path: Union[os.PathLike, str],
    subset: Literal["lizard", "mitosis"],
    split: Literal["train", "val", "test"],
    download: bool = False,
) -> str:
    """Get paths to the lizard-mitosis data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        subset: The choice of data subset.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the stored data.
    """
    if split not in SPLIT_FILES[subset]:
        raise ValueError(f"'{split}' is not a valid split.")

    data_dir = get_lizard_mitosis_data(path, subset, download)
    return _extract_split(data_dir, subset, split)


def get_lizard_mitosis_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    subset: Literal["lizard", "mitosis"],
    split: Literal["train", "val", "test"],
    label_choice: Literal["instances", "semantic"] = "instances",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the lizard-mitosis dataset for nucleus segmentation and classification.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        subset: The choice of data subset.
        split: The choice of data split.
        label_choice: The choice of label type, either instance or semantic (class) labels.
        resize_inputs: Whether to resize the input images.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    data_path = get_lizard_mitosis_paths(path, subset, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=data_path,
        raw_key="raw",
        label_paths=data_path,
        label_key=f"labels/{label_choice}",
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs,
    )


def get_lizard_mitosis_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    subset: Literal["lizard", "mitosis"],
    split: Literal["train", "val", "test"],
    label_choice: Literal["instances", "semantic"] = "instances",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the lizard-mitosis dataloader for nucleus segmentation and classification.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        subset: The choice of data subset.
        split: The choice of data split.
        label_choice: The choice of label type, either instance or semantic (class) labels.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_lizard_mitosis_dataset(
        path, patch_shape, subset, split, label_choice, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
