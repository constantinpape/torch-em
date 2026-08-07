"""The Kromp nuclei dataset contains annotated fluorescence microscopy images of human cells and tissues.

It contains 79 expert-annotated immunofluorescence and DAPI images with 7813 nuclei from normal and cancer
samples. The loader uses the train/test split from https://github.com/kreshuklab/model_ranking.

The curated dataset is located at https://www.ebi.ac.uk/biostudies/studies/S-BIAD634 under the CC0 license.
It was originally published as S-BSST265 and is from https://doi.org/10.1038/s41597-020-00608-w.
Please cite the dataset and publication if you use this dataset in your research.
"""

import os
from zipfile import ZipFile
from typing import List, Literal, Optional, Tuple, Union

from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util


URL = "https://www.ebi.ac.uk/biostudies/files/S-BSST265/dataset.zip"
CHECKSUM = "8285987ed4d57c46a46a55a33c1c085875ea41f429b59cde31d249741aa07ad1"

SAMPLES = {
    "train": (
        "Ganglioneuroblastoma_0",
        "Ganglioneuroblastoma_1",
        "Ganglioneuroblastoma_2",
        "Ganglioneuroblastoma_3",
        "Neuroblastoma_0",
        "Neuroblastoma_1",
        "Neuroblastoma_10",
        "Neuroblastoma_11",
        "Neuroblastoma_2",
        "Neuroblastoma_3",
        "Neuroblastoma_4",
        "Neuroblastoma_5",
        "Neuroblastoma_6",
        "Neuroblastoma_7",
        "Neuroblastoma_8",
        "Neuroblastoma_9",
        "normal_0",
        "normal_1",
        "normal_10",
        "normal_11",
        "normal_12",
        "normal_13",
        "normal_14",
        "normal_15",
        "normal_16",
        "normal_17",
        "normal_18",
        "normal_19",
        "normal_2",
        "normal_20",
        "normal_21",
        "normal_22",
        "normal_23",
        "normal_24",
        "normal_25",
        "normal_3",
        "normal_4",
        "normal_5",
        "normal_6",
        "normal_7",
        "normal_8",
        "normal_9",
    ),
    "test": (
        "Ganglioneuroblastoma_10",
        "Ganglioneuroblastoma_4",
        "Ganglioneuroblastoma_6",
        "Ganglioneuroblastoma_7",
        "Ganglioneuroblastoma_8",
        "Ganglioneuroblastoma_9",
        "Neuroblastoma_12",
        "Neuroblastoma_13",
        "Neuroblastoma_14",
        "Neuroblastoma_15",
        "Neuroblastoma_16",
        "Neuroblastoma_17",
        "normal_26",
        "normal_27",
        "normal_28",
        "normal_29",
        "normal_30",
        "normal_31",
        "normal_32",
        "normal_33",
        "normal_34",
        "normal_35",
        "normal_36",
        "normal_37",
        "normal_38",
        "normal_39",
        "normal_40",
        "otherspecimen_0",
        "otherspecimen_1",
        "otherspecimen_2",
        "otherspecimen_3",
        "otherspecimen_4",
        "otherspecimen_5",
        "otherspecimen_6",
        "otherspecimen_7",
        "otherspecimen_8",
        "otherspecimen_9",
    ),
}


def _get_expected_paths(data_dir: str) -> List[str]:
    samples = SAMPLES["train"] + SAMPLES["test"]
    raw_paths = [os.path.join(data_dir, "rawimages", f"{sample}.tif") for sample in samples]
    label_paths = [os.path.join(data_dir, "groundtruth", f"{sample}.tif") for sample in samples]
    return raw_paths + label_paths


def _extract_data(zip_path: str, data_dir: str) -> None:
    samples = SAMPLES["train"] + SAMPLES["test"]
    with ZipFile(zip_path) as archive:
        for folder in ("rawimages", "groundtruth"):
            for sample in samples:
                archive.extract(f"{folder}/{sample}.tif", data_dir)


def _select_grayscale(raw):
    if raw.ndim == 2:
        return raw
    if raw.ndim == 3 and raw.shape[0] == 3:
        return raw[0]
    raise ValueError(f"Expected a grayscale or RGB Kromp image, got shape {raw.shape}.")


def get_kromp_nuclei_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Kromp nuclei dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data.
    """
    data_dir = os.path.join(path, "kromp_nuclei")
    expected_paths = _get_expected_paths(data_dir)
    if all(os.path.exists(expected_path) for expected_path in expected_paths):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "kromp_nuclei.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    _extract_data(zip_path, data_dir)

    missing_paths = [expected_path for expected_path in expected_paths if not os.path.exists(expected_path)]
    if missing_paths:
        raise RuntimeError(f"Could not find {len(missing_paths)} files after extracting the Kromp nuclei dataset.")
    return data_dir


def get_kromp_nuclei_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Kromp fluorescence images and nucleus instance labels.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The image paths and corresponding label paths.
    """
    if split not in SAMPLES:
        raise ValueError(f"'{split}' is not a valid split. Choose from {list(SAMPLES)}.")

    data_dir = get_kromp_nuclei_data(path, download)
    raw_paths = [os.path.join(data_dir, "rawimages", f"{sample}.tif") for sample in SAMPLES[split]]
    label_paths = [os.path.join(data_dir, "groundtruth", f"{sample}.tif") for sample in SAMPLES[split]]
    return raw_paths, label_paths


def get_kromp_nuclei_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the Kromp nuclei dataset for instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'train' or 'test'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if len(patch_shape) != 2:
        raise ValueError(f"The Kromp nuclei patch shape must be two-dimensional, got {patch_shape}.")

    raw_paths, label_paths = get_kromp_nuclei_paths(path, split, download)
    raw_transform = kwargs.pop("raw_transform", None)
    if raw_transform is None:
        raw_transform = torch_em.transform.get_raw_transform()
    kwargs["raw_transform"] = torch_em.transform.Compose(_select_grayscale, raw_transform, is_multi_tensor=False)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary,
    )
    kwargs = util.ensure_transforms(ndim=2, **kwargs)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs,
    )


def get_kromp_nuclei_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the Kromp nuclei dataloader for instance segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The 2D patch shape to use for training.
        split: The data split. Either 'train' or 'test'.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_kromp_nuclei_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
