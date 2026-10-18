"""The AI4Mat dataset contains microscopy images of materials with semantic mask annotations.

It is a curated collection of seven datasets with predefined train, val and test splits:
- Aachen-Heerlen: bainitic steel microstructures (1,705 images).
- EBC: environmental barrier coatings (24 images).
- EMPS: electron microscopy nanoparticles (465 images).
- Grain: grain structures in stainless steel 316L (816 images).
- MetalDAM: multi-phase metallographic images of advanced high-strength steel (42 images).
- Super: Ni-based superalloy microstructures (23 images).
- UHCS: ultrahigh carbon steel micrographs (550 images).

The dataset is located at https://doi.org/10.5281/zenodo.22039777 (version 1.0) and is distributed under the
CC BY 4.0 license. The datasets were curated from previously published sources, see the Zenodo record for the
references of the respective source datasets. Please cite the Zenodo record and the source datasets
if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://zenodo.org/records/22039777/files"
CHECKSUMS = {
    "Aachen-Heerlen": "24ab2a412e818492a6b9b0fafd86d6bd5dc59e98e422ca12d122558c4f700f8c",
    "EBC": "b870428a07bcd242f6db221b9635ec2fd9f70a4ddb8f566f97b67f2d2773dcd0",
    "EMPS": "1e9c4dca11ae5eda59ac63f65e60be12ffbe08837f746b0b49633d16c12e795d",
    "Grain": "30ca8e80392519bb790d6093b5deb86df4d4aedf8cbe9beedbf329106d54d9d0",
    "MetalDAM": "df0da3dc5690764b6f45feacb3862dfd9a0433e82476c96fbfff6f7725957fa4",
    "Super": "a77726051e1f6c4dc68d0e44193f3ff268a08a767206d0b1eaae066cf101a993",
    "UHCS": "48790928559dea9238f543f6c34912925f6753ffcbb93bfa0cbbd07219001dcc",
}
URLS = {name: f"{BASE_URL}/{name}.zip?download=1" for name in CHECKSUMS}
SPLITS = ("train", "val", "test")


def get_ai4mat_data(
    path: Union[os.PathLike, str],
    name: Literal[tuple(CHECKSUMS)],
    download: bool = False,
) -> str:
    """Download the AI4Mat dataset.

    NOTE: The 'Aachen-Heerlen' archive is 1.2 GB, the other archives are between 7 and 210 MB.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the dataset. One of 'Aachen-Heerlen', 'EBC', 'EMPS', 'Grain', 'MetalDAM', 'Super', 'UHCS'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder with the data of the selected dataset.
    """
    assert name in CHECKSUMS, f"'{name}' is not a valid dataset name. Choose from {list(CHECKSUMS)}."

    data_dir = os.path.join(path, name)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"{name}.zip")
    util.download_source(path=zip_path, url=URLS[name], download=download, checksum=CHECKSUMS[name])
    util.unzip(zip_path=zip_path, dst=path)
    return data_dir


def get_ai4mat_paths(
    path: Union[os.PathLike, str],
    name: Literal[tuple(CHECKSUMS)],
    split: Literal["train", "val", "test"],
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the AI4Mat data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the dataset. One of 'Aachen-Heerlen', 'EBC', 'EMPS', 'Grain', 'MetalDAM', 'Super', 'UHCS'.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    assert split in SPLITS, f"'{split}' is not a valid data split. Choose from {SPLITS}."

    data_dir = get_ai4mat_data(path, name, download)
    image_paths = natsorted(glob(os.path.join(data_dir, split, "images", "*")))
    label_paths = natsorted(glob(os.path.join(data_dir, split, "masks", "*")))
    assert len(image_paths) > 0 and len(image_paths) == len(label_paths)
    return image_paths, label_paths


def get_ai4mat_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    name: Literal[tuple(CHECKSUMS)],
    split: Literal["train", "val", "test"],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AI4Mat dataset for semantic segmentation of materials microscopy images.

    The labels contain the semantic class ids, where 0 is the background.
    The number of classes is 1 for Aachen-Heerlen, EMPS, Grain; 2 for EBC, Super; 4 for MetalDAM; 6 for UHCS.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        name: The name of the dataset. One of 'Aachen-Heerlen', 'EBC', 'EMPS', 'Grain', 'MetalDAM', 'Super', 'UHCS'.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_ai4mat_paths(path, name, split, download)

    kwargs = util.update_kwargs(kwargs, "ndim", 2)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_ai4mat_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    name: Literal[tuple(CHECKSUMS)],
    split: Literal["train", "val", "test"],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DataLoader for semantic segmentation of materials microscopy images in the AI4Mat dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        name: The name of the dataset. One of 'Aachen-Heerlen', 'EBC', 'EMPS', 'Grain', 'MetalDAM', 'Super', 'UHCS'.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ai4mat_dataset(path, patch_shape, name, split, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
