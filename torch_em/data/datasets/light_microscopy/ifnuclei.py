"""The IFNuclei dataset contains annotations for nucleus segmentation
of immuno and DAPI stained fluorescence images.

It contains 79 expert-annotated immunofluorescence and DAPI images with 7813 nuclei from normal and
cancer samples. Pass `split` to use the train and test split of
https://github.com/kreshuklab/model_ranking, which holds 42 and 37 images.

This dataset is from the publication https://doi.org/10.1038/s41597-020-00608-w.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://www.ebi.ac.uk/biostudies/files/S-BSST265/dataset.zip"
CHECKSUM = "8285987ed4d57c46a46a55a33c1c085875ea41f429b59cde31d249741aa07ad1"

SAMPLES = {
    "train": (
        "Ganglioneuroblastoma_0", "Ganglioneuroblastoma_1", "Ganglioneuroblastoma_2",
        "Ganglioneuroblastoma_3", "Neuroblastoma_0", "Neuroblastoma_1",
        "Neuroblastoma_10", "Neuroblastoma_11", "Neuroblastoma_2",
        "Neuroblastoma_3", "Neuroblastoma_4", "Neuroblastoma_5",
        "Neuroblastoma_6", "Neuroblastoma_7", "Neuroblastoma_8",
        "Neuroblastoma_9", "normal_0", "normal_1",
        "normal_10", "normal_11", "normal_12",
        "normal_13", "normal_14", "normal_15",
        "normal_16", "normal_17", "normal_18",
        "normal_19", "normal_2", "normal_20",
        "normal_21", "normal_22", "normal_23",
        "normal_24", "normal_25", "normal_3",
        "normal_4", "normal_5", "normal_6",
        "normal_7", "normal_8", "normal_9",
    ),
    "test": (
        "Ganglioneuroblastoma_10", "Ganglioneuroblastoma_4", "Ganglioneuroblastoma_6",
        "Ganglioneuroblastoma_7", "Ganglioneuroblastoma_8", "Ganglioneuroblastoma_9",
        "Neuroblastoma_12", "Neuroblastoma_13", "Neuroblastoma_14",
        "Neuroblastoma_15", "Neuroblastoma_16", "Neuroblastoma_17",
        "normal_26", "normal_27", "normal_28",
        "normal_29", "normal_30", "normal_31",
        "normal_32", "normal_33", "normal_34",
        "normal_35", "normal_36", "normal_37",
        "normal_38", "normal_39", "normal_40",
        "otherspecimen_0", "otherspecimen_1", "otherspecimen_2",
        "otherspecimen_3", "otherspecimen_4", "otherspecimen_5",
        "otherspecimen_6", "otherspecimen_7", "otherspecimen_8",
        "otherspecimen_9",
    ),
}


def _select_grayscale(raw):
    if raw.ndim == 2:
        return raw
    if raw.ndim == 3 and raw.shape[0] == 3:
        return raw[0]
    raise ValueError(f"Expected a grayscale or RGB IFNuclei image, got shape {raw.shape}.")


def get_ifnuclei_data(path: Union[os.PathLike, str], download: bool = False):
    """Download the IFNuclei dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.
    """
    data_dir = os.path.join(path, "rawimages")
    if os.path.exists(data_dir):
        return

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)


def get_ifnuclei_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "test"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the IFNuclei data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split. Either 'train' or 'test', or None for all images.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_ifnuclei_data(path, download)

    if split is None:
        raw_paths = natsorted(glob(os.path.join(path, "rawimages", "*.tif")))
        label_paths = natsorted(glob(os.path.join(path, "groundtruth", "*")))
    else:
        if split not in SAMPLES:
            raise ValueError(f"'{split}' is not a valid split. Choose from {list(SAMPLES)}, or None for all images.")
        raw_paths = [os.path.join(path, "rawimages", f"{sample}.tif") for sample in SAMPLES[split]]
        label_paths = [os.path.join(path, "groundtruth", f"{sample}.tif") for sample in SAMPLES[split]]
        missing = [p for p in raw_paths + label_paths if not os.path.exists(p)]
        if missing:
            raise RuntimeError(f"Could not find {len(missing)} files of the '{split}' split, e.g. {missing[0]}.")

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_ifnuclei_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "test"]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the IFNuclei dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The data split. Either 'train' or 'test', or None for all images.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_ifnuclei_paths(path, split, download)

    raw_transform = kwargs.pop("raw_transform", None)
    if raw_transform is None:
        raw_transform = torch_em.transform.get_raw_transform()
    kwargs["raw_transform"] = torch_em.transform.Compose(_select_grayscale, raw_transform, is_multi_tensor=False)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary,
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


def get_ifnuclei_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "test"]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the IFNuclei dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split. Either 'train' or 'test', or None for all images.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ifnuclei_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
