"""The AbdomenCT-1K dataset contains annotations for liver, kidney, spleen and pancreas segmentation in CT scans.

The dataset consists of 1112 abdominal CT volumes from 12 medical centers (collected from LiTS, KiTS19, MSD Spleen,
MSD Pancreas, NIH Pancreas and Nanjing University). The annotations of 1000 volumes are public, the annotations of
the remaining 112 volumes are held back for the AbdomenCT-1K benchmarks and these volumes are not used here,
so the dataset provides 1000 image / label pairs.
The label ids are: 1: liver, 2: kidney, 3: spleen, 4: pancreas (see `CLASS_IDS`).

The dataset is located at https://zenodo.org/records/5903099 (images, part 1),
https://zenodo.org/records/5903846 (images, part 2) and https://zenodo.org/records/5903769 (images, part 3 and labels).
NOTE: The label archive is a 7z archive, so the '7z' CLI is required to extract it
(install it via 'conda install -c conda-forge p7zip').

This dataset is from the publication https://doi.org/10.1109/TPAMI.2021.3100536.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "AbdomenCT-1K-ImagePart1.zip": "https://zenodo.org/records/5903099/files/AbdomenCT-1K-ImagePart1.zip?download=1",
    "AbdomenCT-1K-ImagePart2.zip": "https://zenodo.org/records/5903846/files/AbdomenCT-1K-ImagePart2.zip?download=1",
    "AbdomenCT-1K-ImagePart3.zip": "https://zenodo.org/records/5903769/files/AbdomenCT-1K-ImagePart3.zip?download=1",
    "Mask.7z": "https://zenodo.org/records/5903769/files/Mask.7z?download=1",
}

CHECKSUMS = {
    "AbdomenCT-1K-ImagePart1.zip": "3d0d8edd2a8777f8807c6715f633e2fea104f16a8514572664021ed88f97cd03",
    "AbdomenCT-1K-ImagePart2.zip": "3178bd3021dfed0440c4dc269b7f3125b27b60aec96e81b210d7c4c55d0fe894",
    "AbdomenCT-1K-ImagePart3.zip": "76bf789589a13ca1aef9b9c8b771f05c165e6d2013284bbfdb4c4656d4931c30",
    "Mask.7z": "cefa7cab51fb0781876550b032278ca6a1ed2a8c776c8567410ea2b2a6984e50",
}

CLASS_NAMES = ["liver", "kidney", "spleen", "pancreas"]
"""The organs of the AbdomenCT-1K dataset. The label id of an organ is its 1-based index."""

CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
"""Mapping from the organ name to its label id."""


def get_abdomenct_1k_data(path: Union[os.PathLike, str], download: bool = False):
    """Download the AbdomenCT-1K dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.
    """
    os.makedirs(path, exist_ok=True)

    for name, url in URLS.items():
        archive_path = os.path.join(path, name)
        data_dir = os.path.join(path, name.split(".")[0])
        if os.path.exists(data_dir):
            continue

        util.download_source(path=archive_path, url=url, download=download, checksum=CHECKSUMS[name])
        if name.endswith(".zip"):
            util.unzip(zip_path=archive_path, dst=path, remove=False)
        else:
            util.unzip_7z(path_7z=archive_path, dst=data_dir, remove=False)


def get_abdomenct_1k_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the AbdomenCT-1K data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_abdomenct_1k_data(path, download)

    # Only the cases with public annotations are used.
    label_paths = natsorted(glob(os.path.join(path, "Mask", "Case_*.nii.gz")))
    raw_paths = []
    for label_path in label_paths:
        case_id = os.path.basename(label_path)[:-len(".nii.gz")]
        image_paths = glob(os.path.join(path, "AbdomenCT-1K-ImagePart*", f"{case_id}_0000.nii.gz"))
        if len(image_paths) != 1:
            raise RuntimeError(f"Could not find a unique image for '{label_path}', found {image_paths}.")
        raw_paths.append(image_paths[0])

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    return raw_paths, label_paths


def get_abdomenct_1k_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AbdomenCT-1K dataset for abdominal organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_abdomenct_1k_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_abdomenct_1k_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the AbdomenCT-1K dataloader for abdominal organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_abdomenct_1k_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
