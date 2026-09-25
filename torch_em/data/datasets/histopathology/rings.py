"""The RINGS dataset contains annotations for prostate gland and tumor region
segmentation in H&E stained prostate histopathological images.

The dataset is located at https://data.mendeley.com/datasets/h8bdwrtnr5/1.
This dataset is from the publication https://doi.org/10.1016/j.artmed.2021.102076.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "train": "https://data.mendeley.com/public-files/datasets/h8bdwrtnr5/files/8416eb6b-d1c8-4a8c-96fd-2f36a2768e46/file_downloaded",  # noqa
    "test": "https://data.mendeley.com/public-files/datasets/h8bdwrtnr5/files/5846b131-09fa-44c5-afa8-8fbc52adbd88/file_downloaded",  # noqa
}
CHECKSUMS = {
    "train": "af426249bd96d36e2c5e0110d42ceb67a8ebb79d94e2b2c15f4e727ebca38329",
    "test": "f8134f01ce4cbcfd703bf96a8501e6267fe87a66301e15749dac463742f8958d",
}


def get_rings_data(path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False) -> str:
    """Download the RINGS dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where the split data is stored.
    """
    if split not in URLS:
        raise ValueError(f"'{split}' is not a valid split choice.")

    data_dir = os.path.join(path, split.upper())
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"{split}.zip")
    util.download_source(path=zip_path, url=URLS[split], download=download, checksum=CHECKSUMS[split])
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_rings_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    label_choice: Literal["glands", "tumor"] = "glands",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the RINGS data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        label_choice: The segmentation target. Either 'glands' for gland segmentation
            or 'tumor' for tumor region segmentation.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if label_choice not in ("glands", "tumor"):
        raise ValueError(f"'{label_choice}' is not a valid label choice.")

    data_dir = get_rings_data(path, split, download)
    raw_paths = natsorted(glob(os.path.join(data_dir, "IMAGES", "*.png")))

    label_folder = "MANUAL GLANDS" if label_choice == "glands" else "MANUAL TUMOR"
    label_paths = natsorted(glob(os.path.join(data_dir, label_folder, "*.png")))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(
        os.path.basename(raw_path) == os.path.basename(label_path)
        for raw_path, label_path in zip(raw_paths, label_paths)
    )

    return raw_paths, label_paths


def get_rings_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    label_choice: Literal["glands", "tumor"] = "glands",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the RINGS dataset for prostate gland or tumor region segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        label_choice: The segmentation target. Either 'glands' for gland segmentation
            or 'tumor' for tumor region segmentation.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_rings_paths(path, split, label_choice, download)

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
        ndim=2,
        with_channels=True,
        **kwargs,
    )


def get_rings_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    label_choice: Literal["glands", "tumor"] = "glands",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the RINGS dataloader for prostate gland or tumor region segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        label_choice: The segmentation target. Either 'glands' for gland segmentation
            or 'tumor' for tumor region segmentation.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_rings_dataset(path, patch_shape, split, label_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
