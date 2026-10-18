"""The ELMI Mito dataset contains annotations for mitochondria instance segmentation in transmission electron
microscopy (TEM) images.

It consists of five large images (about 2000 x 2000 pixels each) with manually annotated mitochondria instances.
The images were shared by the authors as part of the training data for a MitoNet (empanada) model that was fine-tuned
on mitochondria in these ELMI images. In total the download size is ~40 MB.

The data is available at https://owncloud.cesnet.cz/index.php/s/GRxauIeDbnSLiZV.
The share does not state a licence or a publication to cite, and does not explain the abbreviation 'ELMI'.
Please check with the data providers before using this dataset in your research.
The share also contains the fine-tuned model and its predictions on the images, which are not downloaded here.
"""

import os
from typing import List, Tuple, Union

from natsort import natsorted
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://owncloud.cesnet.cz/index.php/s/GRxauIeDbnSLiZV/download"
IMAGE_NAMES = [f"ELMI_0{i}.tif" for i in range(1, 6)]
CHECKSUMS = {
    "images": {
        "ELMI_01.tif": "17dd6971fb03566cd32dfbf3395640459d9f982195f7ade92a8169f8b0033090",
        "ELMI_02.tif": "dbe01aa9773aaf25898282513f35fc352e8f2e6eb11fa917e7c31c4a4bd8b982",
        "ELMI_03.tif": "e9ea36b449abf9126e5dfc057e9f11b3aa5f114f571e8cec5bc5225dccfb79f8",
        "ELMI_04.tif": "43ac3a04f26c845872ade63efdd6d8d63488d589546b4240576cef2d80aa0a40",
        "ELMI_05.tif": "e6005e1ee2a96a2205bd7c2883cfb25111c9c7434abb563d3e2e3e1b002c2d78",
    },
    "labels": {
        "ELMI_01.tif": "6844b7f078446c351ab0cb3149a479c9183eb5ca1381d9a5b574753fd8719035",
        "ELMI_02.tif": "17d7f6a85ca536770fd65e696e06d6898c6fe5dc7ca45b945952218f90334021",
        "ELMI_03.tif": "74c82d0d5c735dcc0d8955dd462420d4760dd8cb23991bc4fba343727763a597",
        "ELMI_04.tif": "144f3fd0ddd4f19780295c48454d88fd9362871cabec78b45d40c874b30bf213",
        "ELMI_05.tif": "65de9d6839ae70703cc2c2cb72b6ff864db60632b0457fa899825f0df799c6e5",
    },
}
# The folders of the images and the labels on the share.
FOLDERS = {"images": "ELMI_orig_images", "labels": "ELMI_manual_annotation"}


def get_elmi_mito_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the ELMI Mito dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the data.
    """
    for kind, folder in FOLDERS.items():
        os.makedirs(os.path.join(path, folder), exist_ok=True)
        for name in IMAGE_NAMES:
            url = f"{BASE_URL}?path=%2FELMI_images%2F{folder}&files={name}"
            util.download_source(os.path.join(path, folder, name), url, download, checksum=CHECKSUMS[kind][name])

    return path


def get_elmi_mito_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the ELMI Mito data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_elmi_mito_data(path, download)
    raw_paths = natsorted(os.path.join(path, FOLDERS["images"], name) for name in IMAGE_NAMES)
    label_paths = natsorted(os.path.join(path, FOLDERS["labels"], name) for name in IMAGE_NAMES)
    return raw_paths, label_paths


def _binary_labels(labels):
    return (labels > 0).astype("uint8")


def get_elmi_mito_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the ELMI Mito dataset for mitochondria segmentation in TEM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        binary: Whether to return a binary foreground mask. Otherwise the mitochondria instances are returned.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_elmi_mito_paths(path, download)

    if binary:
        kwargs = util.update_kwargs(kwargs, "label_transform", _binary_labels)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_elmi_mito_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the ELMI Mito dataloader for mitochondria segmentation in TEM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        binary: Whether to return a binary foreground mask. Otherwise the mitochondria instances are returned.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_elmi_mito_dataset(path, patch_shape, binary, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
