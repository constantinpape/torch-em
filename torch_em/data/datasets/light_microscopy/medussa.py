"""The MeDuSSA dataset contains annotations for bacterial membrane
instance segmentation in fluorescence microscopy images stained with FM 4-64.

The dataset provides 143 training images and 16 benchmarking images of
membrane-stained bacteria (primarily Bacillus subtilis PY79) with corresponding
instance segmentation masks annotated using JFilament in FIJI.

The dataset is located at https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2350.
This dataset is from the publication https://doi.org/10.1101/2025.10.26.684635.
Please cite it if you use this dataset in your research.
"""

import os
import json
from glob import glob
from typing import Union, Tuple, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://www.ebi.ac.uk/biostudies/files/S-BIAD2350"

SPLIT_FILE_LISTS = {
    "train": {
        "images": "submission_segmentation_training_images_raw.json",
        "masks": "submission_segmentation_training_masks.json",
    },
    "test": {
        "images": "submission_segmentation_benchmarking_images_raw.json",
        "masks": "submission_segmentation_benchmarking_masks.json",
    },
}


def _download_file_lists(path, split):
    """Download and parse JSON file lists from BioStudies to get relative file paths."""
    file_list_dir = os.path.join(path, "file_lists")
    os.makedirs(file_list_dir, exist_ok=True)

    result = {}
    for key in ("images", "masks"):
        json_fname = SPLIT_FILE_LISTS[split][key]
        json_path = os.path.join(file_list_dir, json_fname)

        if not os.path.exists(json_path):
            url = f"{BASE_URL}/{json_fname}"
            util.download_source(path=json_path, url=url, download=True, checksum=None)

        with open(json_path) as f:
            data = json.load(f)

        result[key] = sorted([entry["path"] for entry in data])

    return result["images"], result["masks"]


def _create_h5_data(path, split, image_paths_rel, mask_paths_rel):
    """Create h5 files with raw images and instance labels."""
    import h5py
    import imageio.v3 as imageio
    from tqdm import tqdm

    h5_dir = os.path.join(path, "h5_data", split)
    os.makedirs(h5_dir, exist_ok=True)

    assert len(image_paths_rel) == len(mask_paths_rel), \
        f"Mismatch: {len(image_paths_rel)} images vs {len(mask_paths_rel)} masks for split '{split}'"

    for img_rel, mask_rel in tqdm(
        zip(image_paths_rel, mask_paths_rel),
        total=len(image_paths_rel),
        desc=f"Creating h5 files for '{split}'"
    ):
        fname = os.path.splitext(os.path.basename(img_rel))[0]
        h5_path = os.path.join(h5_dir, f"{fname}.h5")

        if os.path.exists(h5_path):
            continue

        raw = imageio.imread(os.path.join(path, img_rel))
        labels = imageio.imread(os.path.join(path, mask_rel))

        # Handle potential multi-dimensional images (e.g. Z-stacks not fully max-projected).
        if raw.ndim > 2:
            raw = raw.max(axis=0)

        if labels.ndim > 2:
            labels = labels.max(axis=0)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels.astype("int64"), compression="gzip")

    return h5_dir


def get_medussa_data(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"] = "train",
    download: bool = False,
) -> str:
    """Download the MeDuSSA dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. One of 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the directory with the downloaded data.
    """
    assert split in ("train", "test"), f"'{split}' is not a valid split."

    image_paths_rel, mask_paths_rel = _download_file_lists(path, split)

    for rel_path in image_paths_rel + mask_paths_rel:
        local_path = os.path.join(path, rel_path)
        if os.path.exists(local_path):
            continue

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        url = f"{BASE_URL}/{rel_path}"
        util.download_source(path=local_path, url=url, download=download, checksum=None)

    return path


def get_medussa_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"] = "train",
    download: bool = False,
) -> List[str]:
    """Get paths to the MeDuSSA data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. One of 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the h5 data.
    """
    from natsort import natsorted

    get_medussa_data(path, split, download)

    h5_dir = os.path.join(path, "h5_data", split)
    if not os.path.exists(h5_dir) or len(glob(os.path.join(h5_dir, "*.h5"))) == 0:
        image_paths_rel, mask_paths_rel = _download_file_lists(path, split)
        _create_h5_data(path, split, image_paths_rel, mask_paths_rel)

    h5_paths = natsorted(glob(os.path.join(h5_dir, "*.h5")))
    assert len(h5_paths) > 0, f"No data found for split '{split}'"

    return h5_paths


def get_medussa_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MeDuSSA dataset for bacterial membrane segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    h5_paths = get_medussa_paths(path, split, download)

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


def get_medussa_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MeDuSSA dataloader for bacterial membrane segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. One of 'train' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_medussa_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
