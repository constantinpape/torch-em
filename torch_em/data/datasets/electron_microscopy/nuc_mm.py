"""NucMM is a dataset for the segmentation of nuclei in EM and X-Ray.

This dataset is from the publication https://doi.org/10.1007/978-3-030-87193-2_16.
Please cite it if you use this dataset for a publication.
"""


import os
from glob import glob
from typing import Tuple, Union

import torch_em

from torch.utils.data import Dataset, DataLoader

from .. import util

URL = "https://drive.google.com/drive/folders/1_4CrlYvzx0ITnGlJOHdgcTRgeSkm9wT8"


def _extract_split(image_folder, label_folder, output_folder):
    import h5py

    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted(glob(os.path.join(image_folder, "*.h5")))
    label_files = sorted(glob(os.path.join(label_folder, "*.h5")))
    assert len(image_files) == len(label_files)
    for image, label in zip(image_files, label_files):
        with h5py.File(image, "r") as f:
            vol = f["main"][:]
        with h5py.File(label, "r") as f:
            seg = f["main"][:]
        assert vol.shape == seg.shape
        out_path = os.path.join(output_folder, os.path.basename(image))
        with h5py.File(out_path, "a") as f:
            f.create_dataset("raw", data=vol, compression="gzip")
            f.create_dataset("labels", data=seg, compression="gzip")


def get_nuc_mm_data(path: Union[os.PathLike, str], sample: str, download: bool) -> str:
    """Download the NucMM training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        sample: The NucMM samples to use. The available samples are 'mouse' and 'zebrafish'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the training data.
    """
    assert sample in ("mouse", "zebrafish")

    sample_folder = os.path.join(path, sample)
    if os.path.exists(sample_folder):
        return sample_folder

    # Downloading the dataset
    util.download_source_gdrive(path, URL, download, download_type="folder")

    if sample == "mouse":
        input_folder = os.path.join(path, "Mouse (NucMM-M)")
    else:
        input_folder = os.path.join(path, "Zebrafish (NucMM-Z)")
    assert os.path.exists(input_folder), input_folder

    sample_folder = os.path.join(path, sample)
    _extract_split(
        os.path.join(input_folder, "Image", "train"), os.path.join(input_folder, "Label", "train"),
        os.path.join(sample_folder, "train")
    )
    _extract_split(
        os.path.join(input_folder, "Image", "val"), os.path.join(input_folder, "Label", "val"),
        os.path.join(sample_folder, "val")
    )
    return sample_folder


def get_nuc_mm_dataset(
    path: Union[os.PathLike, str],
    sample: str,
    split: str,
    patch_shape: Tuple[int, int, int],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the NucMM dataset for the segmentation of nuclei in X-Ray and EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        sample: The CREMI samples to use. The available samples are 'A', 'B', 'C'.
        split: The split for the dataset, either 'train' or 'val'.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    assert split in ("train", "val")

    sample_folder = get_nuc_mm_data(path, sample, download)
    split_folder = os.path.join(sample_folder, split)
    paths = sorted(glob(os.path.join(split_folder, "*.h5")))

    raw_key, label_key = "raw", "labels"
    return torch_em.default_segmentation_dataset(
        paths, raw_key, paths, label_key, patch_shape, is_seg_dataset=True, **kwargs
    )


def get_nuc_mm_loader(
    path: Union[os.PathLike, str],
    sample: str,
    split: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the NucMM dataset for the segmentation of nuclei in X-Ray and EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        sample: The CREMI samples to use. The available samples are 'A', 'B', 'C'.
        split: The split for the dataset, either 'train' or 'val'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
       The segmentation dataset.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_nuc_mm_dataset(path, sample, split, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
