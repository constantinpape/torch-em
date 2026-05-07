"""AxomEM is a datast for segmenting axons in electron microscopy.
It contains two large annotated volumes, one from mouse cortex, the other from human cortex.
This dataset was used for the AxonEM Challenge: https://axonem.grand-challenge.org/.

Please cite the publication https://arxiv.org/abs/2107.05451 if you use this dataset for your research.
"""

import os
from glob import glob
from typing import Union, Sequence, List, Tuple

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "human": "https://huggingface.co/datasets/pytc/AxonEM/resolve/main/EM30-H-train-9vol-pad-20-512-512.zip",
    "mouse": "https://huggingface.co/datasets/pytc/AxonEM/resolve/main/EM30-M-train-9vol-pad-20-512-512.zip",
}

CHECKSUMS = {
    "human": "0b53d155ff62f5e24c552bf90adce329fcf9a8fefd5c697f8bcd0312a61fda60",
    "mouse": "dae06b5dabe388ab7a0ff4e51548174f041a338d0d06bd665586aa7fdd43bac2",
}


def get_axonem_data(path: Union[os.PathLike, str], samples: Sequence[str], download: bool = False):
    """Download the AxonEM training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        samples: The samples to download. The available samples are 'human' and 'mouse'.
        download: Whether to download the data if it is not present.
    """
    if isinstance(samples, str):
        samples = [samples]

    assert len(set(samples) - {"human", "mouse"}) == 0, f"{samples}"
    os.makedirs(path, exist_ok=True)

    for sample in samples:
        dst = os.path.join(path, sample)
        if os.path.exists(dst):
            continue

        os.makedirs(dst, exist_ok=True)

        # Download the zipfile.
        zip_path = os.path.join(path, f"{sample}.zip")
        util.download_source(path=zip_path, url=URLS[sample], download=download, checksum=CHECKSUMS[sample])

        # Extract the h5 crops from the zipfile.
        util.unzip(zip_path=zip_path, dst=dst, remove=True)

        if sample == "mouse":
            # NOTE: We need to make a hotfix by removing a crop which does not have masks.
            label_path = os.path.join(path, "mouse", "valid_mask.h5")
            os.remove(label_path)

            # And the additional volume with no corresponding mask.
            image_path = os.path.join(path, "mouse", "im_675-800-800_pad.h5")
            os.remove(image_path)


def get_axonem_paths(
    path: Union[os.PathLike, str], samples: Sequence[str], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths for the AxonEM training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        samples: The samples to download. The available samples are 'human' and 'mouse'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image volumes.
        List of filepaths for the label volumes.
    """
    get_axonem_data(path, samples, download)

    if isinstance(samples, str):
        samples = [samples]

    image_paths, label_paths = [], []
    for sample in samples:
        curr_image_paths = glob(os.path.join(path, sample, "im_*.h5"))
        image_paths.extend(curr_image_paths)
        label_paths.extend([p.replace("im_", "seg_") for p in curr_image_paths])

    return image_paths, label_paths


def get_axonem_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    samples: Sequence[str] = ("human", "mouse"),
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AxonEM dataset for the segmentation of axons in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        samples: The samples to download. The available samples are 'human' and 'mouse'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    image_paths, label_paths = get_axonem_paths(path, samples, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="main",
        label_paths=label_paths,
        label_key="main",
        patch_shape=patch_shape,
        **kwargs
    )


def get_axonem_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    samples: Sequence[str] = ("human", "mouse"),
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the AxonEM dataloader for the segmentation of axons in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        samples: The samples to download. The available samples are 'human' and 'mouse'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_axonem_dataset(path, patch_shape, samples, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
