"""The PU2756 dataset contains annotations for pulmonary tumor segmentation in B-mode
lung ultrasound images.

The dataset contains 2,756 ultrasound images of peripheral pulmonary lesions from 2,756
unique patients, with expert sonographer pixel-level tumor masks and biopsy-confirmed
benign / malignant pathology labels.

This dataset is located at https://doi.org/10.6084/m9.figshare.32672274.v1 (CC BY 4.0).
This dataset is from the publication https://doi.org/10.1038/s41597-026-07715-0.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "images": "https://ndownloader.figshare.com/files/65547621",
    "masks": "https://ndownloader.figshare.com/files/65547618",
}
CHECKSUMS = {
    "images": "f91821641147cbc4d2fd25a5e590e7028b8b1ec5a368987866f2c5551a011a2e",
    "masks": "3e88ec64da10555b70c170d31265a8474570df6592c23ec4906a57c89f78a7e3",
}


def get_pu2756_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PU2756 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "masks")
    if os.path.exists(data_dir):
        return path

    os.makedirs(path, exist_ok=True)

    for name, url in URLS.items():
        zip_path = os.path.join(path, f"{name}.zip")
        util.download_source(path=zip_path, url=url, download=download, checksum=CHECKSUMS[name])
        util.unzip(zip_path=zip_path, dst=path)

    return path


def get_pu2756_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the PU2756 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_pu2756_data(path=path, download=download)

    image_paths = sorted(glob(os.path.join(data_dir, "images", "*", "*.png")))
    gt_paths = sorted(glob(os.path.join(data_dir, "masks", "*", "*.png")))

    if len(image_paths) == 0 or len(image_paths) != len(gt_paths):
        raise RuntimeError("Something went wrong with fetching the image and label paths.")

    return image_paths, gt_paths


def get_pu2756_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PU2756 dataset for pulmonary tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_pu2756_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_pu2756_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PU2756 dataloader for pulmonary tumor segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_pu2756_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
