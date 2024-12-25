"""The PENGWIN dataset contains annotation for pelvic bone fracture and
fragments in CT and X-Ray images.

This dataset is from the challenge: https://pengwin.grand-challenge.org/pengwin/.
This dataset is related to the publication: https://doi.org/10.1007/978-3-031-43996-4_30.
Please cite them if you use this dataset for your publication.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "CT": [
        "https://zenodo.org/records/10927452/files/PENGWIN_CT_train_images_part1.zip",  # inputs part 1
        "https://zenodo.org/records/10927452/files/PENGWIN_CT_train_images_part2.zip",  # inputs part 2
        "https://zenodo.org/records/10927452/files/PENGWIN_CT_train_labels.zip",  # labels
    ],
    "X-Ray": ["https://zenodo.org/records/10913196/files/train.zip"]
}

CHECKSUMS = {
    "CT": [
        "e2e9f99798960607ffced1fbdeee75a626c41bf859eaf4125029a38fac6b7609",  # inputs part 1
        "19f3cdc5edd1daf9324c70f8ba683eed054f6ed8f2b1cc59dbd80724f8f0bbb2",  # inputs part 2
        "c4d3857e02d3ee5d0df6c8c918dd3cf5a7c9419135f1ec089b78215f37c6665c"  # labels
    ],
    "X-Ray": ["48d107979eb929a3c61da4e75566306a066408954cf132907bda570f2a7de725"]
}

TARGET_DIRS = {
    "CT": ["CT/images", "CT/images", "CT/labels"],
    "X-Ray": ["X-Ray"]
}

MODALITIES = ["CT", "X-Ray"]


def get_pengwin_data(
    path: Union[os.PathLike, str], modality: Literal["CT", "X-Ray"], download: bool = False
) -> str:
    """Download the PENGWIN dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The choice of modality for inputs.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downlaoded.
    """
    if not isinstance(modality, str) and modality in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Please choose from {MODALITIES}.")

    data_dir = os.path.join(path, "data")
    if os.path.exists(os.path.join(data_dir, modality)):
        return data_dir

    os.makedirs(path, exist_ok=True)

    for url, checksum, dst_dir in zip(URLS[modality], CHECKSUMS[modality], TARGET_DIRS[modality]):
        zip_path = os.path.join(path, os.path.split(url)[-1])
        util.download_source(path=zip_path, url=url, download=download, checksum=checksum)
        util.unzip(zip_path=zip_path, dst=os.path.join(data_dir, dst_dir))

    return data_dir


def get_pengwin_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'val', 'test'],
    modality: Literal["CT", "X-Ray"],
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the PENGWIN data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        modality: The choice of modality for inputs.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_pengwin_data(path, modality, download)

    if modality == "CT":
        image_paths = natsorted(glob(os.path.join(data_dir, modality, "images", "*.mha")))
        gt_paths = natsorted(glob(os.path.join(data_dir, modality, "labels", "*.mha")))
    else:  # X-Ray
        base_dir = os.path.join(data_dir, modality, "train")
        image_paths = natsorted(glob(os.path.join(base_dir, "input", "images", "*.tif")))
        gt_paths = natsorted(glob(os.path.join(base_dir, "output", "images", "*.tif")))

    print(len(image_paths))

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0

    return image_paths, gt_paths


def get_pengwin_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    modality: Literal["CT", "X-Ray"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PENGWIN dataset for pelvic fracture segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        modality: The choice of modality for inputs.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_pengwin_paths(path, split, modality, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs
    )


def get_pengwin_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    modality: Literal["CT", "X-Ray"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PENGWIN dataloader for pelvic fracture segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        modality: The choice of modality for inputs.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pengwin_dataset(path, patch_shape, split, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
