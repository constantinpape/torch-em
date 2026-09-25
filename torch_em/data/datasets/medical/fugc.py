"""The FUGC dataset contains annotations for cervical lip segmentation in transvaginal ultrasound images.

The dataset consists of 890 transvaginal ultrasound images (336x544 pixels, RGB) from the Fetal Ultrasound
Grand Challenge on semi-supervised cervical segmentation. The anterior and posterior cervical lips were
segmented automatically and then manually corrected by an experienced radiologist.

Labeled images are available for three splits: 'train' (50), 'val' (90) and 'test' (300). The training split
additionally contains 450 unlabeled images, which are not used by this loader. The label ids are 0 (background),
1 (anterior lip) and 2 (posterior lip). The record only names the two lips without listing the ids, so this
order is inferred: the structure with id 1 lies above the one with id 2 in all 140 labeled training and
validation images.

The data is located at https://doi.org/10.5281/zenodo.16893174, released under a CC-BY-4.0 license.

Please cite the Zenodo record if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/16893174/files/FUGC%20(Dataset).zip"
CHECKSUM = "e4dea63dd744885835c9ce8035a640a6ae42780ae21180601ddb09590fa73ce4"

SPLITS = {"train": os.path.join("train", "labeled_data"), "val": "val", "test": "test"}


def get_fugc_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the FUGC dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "FUGC (Dataset)", "dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "FUGC.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    assert os.path.exists(data_dir), f"The extraction of the FUGC archive did not create '{data_dir}'."

    return data_dir


def get_fugc_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the FUGC data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. One of 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if split not in SPLITS:
        raise ValueError(f"'{split}' is not a valid split. Choose one of {list(SPLITS)}.")

    data_dir = get_fugc_data(path, download)

    label_paths = natsorted(glob(os.path.join(data_dir, SPLITS[split], "labels", "*.png")))
    raw_paths = [os.path.join(data_dir, SPLITS[split], "images", os.path.basename(p)) for p in label_paths]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(os.path.exists(p) for p in raw_paths)

    return raw_paths, label_paths


def get_fugc_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the FUGC dataset for cervical lip segmentation in transvaginal ultrasound.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. One of 'train', 'val' or 'test'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_fugc_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_fugc_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the FUGC dataloader for cervical lip segmentation in transvaginal ultrasound.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. One of 'train', 'val' or 'test'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_fugc_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
