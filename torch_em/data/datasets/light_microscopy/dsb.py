"""This Dataset was used in a Kaggle Data Science Bowl. It contains light microscopy
images with annotations for nucleus segmentation.

NOTE:
- The 'full' dataset has been taken from https://github.com/ibmua/data-science-bowl-2018-train-set,
as recommended in BBBC website: https://bbbc.broadinstitute.org/BBBC038.
- The 'reduced' dataset is the fluorescence image set from StarDist.

The dataset is described in the publication https://doi.org/10.1038/s41592-019-0612-7.
Please cite it if you use this dataset in your research.
"""

import os
import shutil
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import List, Optional, Tuple, Union, Literal

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util
from .neurips_cell_seg import to_rgb


DSB_URLS = {
    "full": "https://github.com/ibmua/data-science-bowl-2018-train-set/raw/master/train-hand.zip",
    "reduced": "https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip"
}
CHECKSUMS = {
    "full": "d218b8706cd7b9a2d7171268a6e99c7b0e94605af46521ff2ffd5a17708b1af6",
    "reduced": "e44921950edce378063aa4457e625581ba35b4c2dbd9a07c19d48900129f386f"
}


def _merge_instances(path):
    for id_path in tqdm(glob(os.path.join(path, "full", "*")), desc="Preprocessing labels"):
        id = os.path.basename(id_path)

        # Let's preprocess the image: remove alpha channel and make distinction of histopatho vs fluo images.
        image = imageio.imread(os.path.join(id_path, "images", f"{id}.png"))
        assert image.ndim == 3 and image.shape[-1] == 4, image.shape

        image = image[..., :-1]  # Remove alpha channel
        r, g, b = image.transpose(2, 0, 1)
        if np.array_equal(r, g) and np.array_equal(g, b):
            dname = "fluo"
            # Store only one channel for fluorescence images.
            imageio.imwrite(os.path.join(id_path, "images", f"{dname}_{id}.png"), image[..., -1], compression="zlib")
        else:
            dname = "histopatho"
            # Store all three channels for histopathology images.
            imageio.imwrite(os.path.join(id_path, "images", f"{dname}_{id}.png"), image, compression="zlib")

        os.remove(os.path.join(id_path, "images", f"{id}.png"))

        # Next, let's merge the instances.
        label_paths = glob(os.path.join(id_path, "masks", "*"))
        shape = imageio.imread(label_paths[0]).shape

        instances = np.zeros(shape)
        for i, lpath in enumerate(label_paths, start=1):
            instances[imageio.imread(lpath) > 0] = i

        os.makedirs(os.path.join(id_path, "preprocessed_labels"))
        imageio.imwrite(
            os.path.join(id_path, "preprocessed_labels", f"{dname}_{id}.tif"),
            instances.astype("uint32"),
            compression="zlib"
        )
        shutil.rmtree(os.path.join(id_path, "masks"))  # Removing per-object masks after storing merged instances.


def get_dsb_data(path: Union[os.PathLike, str], source: Literal["full", "reduced"], download: bool):
    """Download the DSB training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        source: The source of the dataset. Can either be 'full' for the complete dataset,
            or 'reduced' for the dataset excluding histopathology images.
        download: Whether to download the data if it is not present.
    """
    if source not in DSB_URLS.keys():
        raise ValueError(f"'{source}' is not a valid data source.")

    train_out_path = os.path.join(path, "train")
    test_out_path = os.path.join(path, "test")
    if source == "reduced" and os.path.exists(train_out_path) and os.path.exists(test_out_path):
        return

    full_out_path = os.path.join(path, "full")
    if source == "full" and os.path.exists(full_out_path):
        return

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "dsb.zip" if source == "reduced" else "train-hand.zip")
    util.download_source(zip_path, DSB_URLS[source], download, CHECKSUMS[source])
    util.unzip(zip_path, path, True)

    if source == "reduced":
        shutil.move(os.path.join(path, "dsb2018", "train"), train_out_path)
        shutil.move(os.path.join(path, "dsb2018", "test"), test_out_path)
    else:
        shutil.move(os.path.join(path, "train-hand"), os.path.join(path, "full"))
        _merge_instances(path)


def get_dsb_paths(
    path: Union[os.PathLike, str],
    source: Literal["full", "reduced"],
    split: Optional[Literal["train", "test"]] = None,
    domain: Optional[Literal["fluo", "histopatho"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the DSB data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        source: The source of the dataset. Can either be 'full' for the complete dataset,
            or 'reduced' for the dataset excluding histopathology images.
        split: The split to use for the dataset. Either 'train' or 'test'.
        domain: The choice of modality in dataset.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the folder where the images are stored.
        List of filepaths for the folder where the labels are stored.
    """
    get_dsb_data(path, source, download)

    if source == "reduced":
        if domain is not None:
            assert domain in "fluo", "The reduced set only has 'fluo' images."

        if split is None:
            split = "t*"  # reduced set returns all "train" and "test" sets if split is None.

        raw_paths = natsorted(glob(os.path.join(path, split, "images", "*.tif")))
        label_paths = natsorted(glob(os.path.join(path, split, "masks", "*.tif")))
    else:
        if domain is None:
            domain = "*"

        assert split is None, "There are no splits available for this data."

        raw_paths = natsorted(glob(os.path.join(path, "full", "*", "images", f"{domain}_*.png")))
        label_paths = natsorted(glob(os.path.join(path, "full", "*", "preprocessed_labels", f"{domain}_*.tif")))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_dsb_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    source: Literal["full", "reduced"] = "reduced",
    split: Optional[Literal["train", "test"]] = None,
    domain: Optional[Literal["fluo", "histopatho"]] = None,
    binary: bool = False,
    boundaries: bool = False,
    offsets: Optional[List[List[int]]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DSB dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        source: The source of the dataset. Can either be 'full' for the complete dataset,
            or 'reduced' for the dataset excluding histopathology images.
        split: The split to use for the dataset. Either 'train' or 'test'.
        domain: The choice of modality in dataset.
        binary: Whether to use a binary segmentation target.
        boundaries: Whether to compute boundaries as the target.
        offsets: Offset values for affinity computation used as target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    raw_paths, label_paths = get_dsb_paths(path, source, split, domain, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )
    kwargs = util.update_kwargs(kwargs, "ndim", 2)

    # This is done for when user requests all images in "full" dataset.
    if "raw_transform" not in kwargs and domain is None:
        kwargs["raw_transform"] = torch_em.transform.get_raw_transform(augmentation2=to_rgb)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_dsb_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    source: Literal["full", "reduced"] = "reduced",
    split: Optional[Literal["train", "test"]] = None,
    domain: Optional[Literal["fluo", "histopatho"]] = None,
    binary: bool = False,
    boundaries: bool = False,
    offsets: Optional[List[List[int]]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DSB dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        source: The source of the dataset. Can either be 'full' for the complete dataset,
            or 'reduced' for the dataset excluding histopathology images.
        split: The split to use for the dataset. Either 'train' or 'test'.
        domain: The choice of modality in dataset.
        binary: Whether to use a binary segmentation target.
        boundaries: Whether to compute boundaries as the target.
        offsets: Offset values for affinity computation used as target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_dsb_dataset(
        path, patch_shape, source, split, domain, binary, boundaries, offsets, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
