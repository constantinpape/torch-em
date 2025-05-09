"""The OrganoID dataset contains annotations for pancreatic organoids in brightfield images.

The dataset is from the publication https://doi.org/10.1371/journal.pcbi.1010584.
Please cite it if you use this dataset for your research.
"""

import os
import shutil
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URL = "https://osf.io/download/69nr8/"
CHECKSUM = "a399288524d12bbadeebb38d52711fa746402456257b0cc6531d8c3c5a0cb8f1"


def _store_files_as_h5(data_dir, image_dir, image_pattern, label_dir, label_pattern):

    import h5py

    if os.path.exists(data_dir):
        return

    os.makedirs(data_dir, exist_ok=True)

    image_paths = natsorted(glob(os.path.join(image_dir, image_pattern)))
    gt_paths = natsorted(glob(os.path.join(label_dir, label_pattern)))

    assert image_paths and len(image_paths) == len(gt_paths)

    for image_path, gt_path in zip(image_paths, gt_paths):
        image = imageio.imread(image_path)
        gt = imageio.imread(gt_path)

        if gt.ndim == 3:
            gt = gt[..., 0]  # Choose one label channel as all are same.

        gt = connected_components(gt > 0).astype("uint16")  # Run connected components to get instances.

        # Preprocess the image (ensure all images are 3-channel).
        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :-1]  # Remove alpha channel
        elif image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        assert image.ndim == 3 and image.shape[-1] == 3, image.shape

        # Now, make channels first (to make this work with our dataset)
        image = image.transpose(2, 0, 1)

        with h5py.File(os.path.join(data_dir, f"{Path(image_path).stem}.h5"), "w") as f:
            f.create_dataset(name="raw", data=image, compression="gzip")
            f.create_dataset(name="labels", data=gt, compression="gzip")


def _preprocess_per_species(data_dir, stype, dirname):

    _store_files_as_h5(
        data_dir=os.path.join(data_dir, dirname, "train"),
        image_dir=os.path.join(data_dir, stype, "training", "pre_augmented", "images"),
        image_pattern="*",
        label_dir=os.path.join(data_dir, stype, "training", "pre_augmented", "segmentations"),
        label_pattern="*",
    )

    _store_files_as_h5(
        data_dir=os.path.join(data_dir, dirname, "val"),
        image_dir=os.path.join(data_dir, stype, "validation", "images"), image_pattern="*",
        label_dir=os.path.join(data_dir, stype, "validation", "segmentations"), label_pattern="*",
    )

    _store_files_as_h5(
        data_dir=os.path.join(data_dir, dirname, "test"),
        image_dir=os.path.join(data_dir, stype, "testing", "images"), image_pattern="*",
        label_dir=os.path.join(data_dir, stype, "testing", "segmentations"), label_pattern="*",
    )


def _preprocess_data(data_dir):

    import h5py

    # Let's start assorting the OG PDAC organoids data. We will call this the "original" data.
    print("Preprocessing 'original' data")
    _preprocess_per_species(data_dir, "OriginalData", "original")

    # Next, we go to the 'MouseOrganoids' data. We will call this the "mouse" data.
    print("Preprocessing 'mouse' data")
    _preprocess_per_species(data_dir, "MouseOrganoids", "mouse")

    # And finally, the 'GemcitabineScreen' data. This is a cool data, as the inputs
    # have two channels: BF and PI (propidium iodide), responsible for reporting cellular necrosis.
    # We will call this data as "gemcitabine".
    gdir = os.path.join(data_dir, "gemcitabine")
    if not os.path.exists(gdir):
        print("Preprocessing 'gemcitabine' data")
        os.makedirs(os.path.join(data_dir, "gemcitabine"), exist_ok=True)

        bf_paths = natsorted(glob(os.path.join(data_dir, "GemcitabineScreen", "BF", "*.tif")))
        pi_paths = natsorted(glob(os.path.join(data_dir, "GemcitabineScreen", "PI", "*.tif")))
        label_paths = natsorted(glob(os.path.join(data_dir, "GemcitabineScreen", "OrganoIDProcessed", "*_labeled.tif")))

        assert label_paths and len(label_paths) == len(bf_paths) == len(pi_paths)

        for bf_path, pi_path, label_path in zip(bf_paths, pi_paths, label_paths):
            bf_image = imageio.imread(bf_path)
            pi_image = imageio.imread(pi_path)
            gt = imageio.imread(label_path)

            assert bf_image.shape == pi_image.shape == gt.shape

            with h5py.File(os.path.join(gdir, f"{Path(bf_path).stem}.h5"), "w") as f:
                f.create_dataset(name="raw/bf", data=bf_image, compression="gzip")
                f.create_dataset(name="raw/pi", data=pi_image, compression="gzip")
                f.create_dataset(name="labels", data=gt, compression="gzip")

    # Let's remove all other data folders.
    shutil.rmtree(os.path.join(data_dir, "OriginalData"))
    shutil.rmtree(os.path.join(data_dir, "MouseOrganoids"))
    shutil.rmtree(os.path.join(data_dir, "GemcitabineScreen"))


def get_organoid_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the OrganoID dataset.

    Args:
        path: Filepath to the folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "data.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir, remove=False)

    _preprocess_data(data_dir)

    return data_dir


def get_organoid_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "val", "test"]] = None,
    source: Literal["gemcitabine", "mouse", "original"] = "original",
    download: bool = False,
) -> List[str]:
    """Get paths to the OrganoID data.

    Args:
        path: Filepath to the folder where the downloaded data will be saved.
        split: The data split to use.
        source: The data source to use.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    if source == "gemcitabine":
        assert split is None, "The 'gemcitabine' data has no data splits."
        split = ""
    else:
        assert split is not None, f"The '{source}' data expects a data split to be chosen."

    data_dir = get_organoid_data(path, download)
    input_paths = natsorted(glob(os.path.join(data_dir, source, split, "*.h5")))
    assert input_paths and len(input_paths) > 0
    return input_paths


def get_organoid_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "val", "test"]] = None,
    source: Literal["gemcitabine", "mouse", "original"] = "original",
    source_channels: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get OrganoID dataset for organoid segmentation in brightfield microscopy images.

    Args:
        path: Filepath to the folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split to use.
        source: The data source to use.
        source_channel: The data source channel to use.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    input_paths = get_organoid_paths(path, split, source, download)

    if source == "gemcitabine":
        assert source_channels is not None, "You must choose a 'source_channel' for 'gemcitabine' data."
        raw_key = [source_channels] if not isinstance(str) else source_channels
        with_channels = (len(raw_key) > 1)
    else:
        assert source_channels is None, f"You cannot choose a 'source_channel' for '{source}' data."
        raw_key = "raw"
        with_channels = True

    return torch_em.default_segmentation_dataset(
        raw_paths=input_paths,
        raw_key=raw_key,
        label_paths=input_paths,
        label_key="labels",
        is_seg_dataset=True,
        ndim=2,
        patch_shape=patch_shape,
        with_channels=with_channels,
        **kwargs
    )


def get_organoid_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "val", "test"]] = None,
    source: Literal["gemcitabine", "mouse", "original"] = "original",
    source_channels: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get OrganoID dataloader for organoid segmentation in brightfield microscopy images.

    Args:
        path: Filepath to the folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use.
        source: The data source to use.
        source_channel: The data source channel to use.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_organoid_dataset(path, patch_shape, split, source, source_channels, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
