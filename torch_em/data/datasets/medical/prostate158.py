"""The Prostate158 dataset contains annotations for prostate zone and prostate cancer segmentation
in biparametric 3T MRI.

Each study provides a T2-weighted sequence ('t2'), a diffusion-weighted sequence ('dwi') and the corresponding
apparent diffusion coefficient map ('adc'). All sequences of a study were resampled to the same grid,
so that the labels can be used with any of them.

NOTE: The label legends are described as following:
1: For the anatomical zones ('anatomy', annotated by reader 1 in the T2-weighted sequence):
- background: 0, transition zone (central gland): 1 and peripheral zone: 2.
2: For the prostate cancer lesions ('tumor', annotated by reader 1 in the ADC map):
- background: 0 and tumor: 1. Studies without a lesion have an empty label volume.

The official split provides 119 training and 20 validation studies (record https://zenodo.org/records/6481141)
and 19 test studies with additional annotations from a second reader (record https://zenodo.org/records/6592345).

The dataset is located at https://github.com/kbressem/prostate158.

This dataset is from the publication https://doi.org/10.1016/j.compbiomed.2022.105817.
Please cite it if you use this dataset in your research.
"""

import os
import csv
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "train": "https://zenodo.org/records/6481141/files/prostate158_train.zip?download=1",
    "test": "https://zenodo.org/records/6592345/files/prostate158_test.zip?download=1",
}

CHECKSUMS = {
    "train": "7a97b263be1bdbc79f6c8a3461e010b2cea9a266249af47671616dadff77da2e",
    "test": "d64a7b94f0654de21150af9b7a5734b2921ec4793bc129b98803ef9ae9d459c4",
}

LABEL_COLUMNS = {"anatomy": "t2_anatomy_reader1", "tumor": "adc_tumor_reader1"}


def get_prostate158_data(
    path: Union[os.PathLike, str], split: Literal["train", "valid", "test"], download: bool = False
) -> str:
    """Download the Prostate158 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. The 'train' and 'valid' splits share one archive, 'test' has its own.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the data of the requested split.
    """
    if split not in ("train", "valid", "test"):
        raise ValueError(f"'{split}' is not a valid split.")

    archive = "test" if split == "test" else "train"
    data_dir = os.path.join(path, f"prostate158_{archive}")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"prostate158_{archive}.zip")
    util.download_source(path=zip_path, url=URLS[archive], download=download, checksum=CHECKSUMS[archive])
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_prostate158_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "valid", "test"],
    sequence: Literal["t2", "adc", "dwi"] = "t2",
    label_type: Literal["anatomy", "tumor"] = "anatomy",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Prostate158 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split. Either 'train', 'valid' or 'test'.
        sequence: The MRI sequence to use as input. Either 't2', 'adc' or 'dwi'.
        label_type: The type of annotations. Either 'anatomy' (prostate zones) or 'tumor' (cancer lesions).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if sequence not in ("t2", "adc", "dwi"):
        raise ValueError(f"'{sequence}' is not a valid sequence.")
    if label_type not in LABEL_COLUMNS:
        raise ValueError(f"'{label_type}' is not a valid label type.")

    data_dir = get_prostate158_data(path, split, download)

    # The official csv files list the volumes per study, including empty tumor labels for studies without lesion.
    with open(os.path.join(data_dir, f"{split}.csv"), "r") as f:
        rows = list(csv.DictReader(f))

    raw_paths = [os.path.join(data_dir, row[sequence]) for row in rows]
    label_paths = [os.path.join(data_dir, row[LABEL_COLUMNS[label_type]]) for row in rows]

    missing = [p for p in raw_paths + label_paths if not os.path.exists(p)]
    assert len(missing) == 0, f"The following files are missing: {missing}"

    return raw_paths, label_paths


def get_prostate158_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "valid", "test"],
    sequence: Literal["t2", "adc", "dwi"] = "t2",
    label_type: Literal["anatomy", "tumor"] = "anatomy",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Prostate158 dataset for prostate zone and prostate cancer segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'valid' or 'test'.
        sequence: The MRI sequence to use as input. Either 't2', 'adc' or 'dwi'.
        label_type: The type of annotations. Either 'anatomy' (prostate zones) or 'tumor' (cancer lesions).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_prostate158_paths(path, split, sequence, label_type, download)

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


def get_prostate158_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "valid", "test"],
    sequence: Literal["t2", "adc", "dwi"] = "t2",
    label_type: Literal["anatomy", "tumor"] = "anatomy",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Prostate158 dataloader for prostate zone and prostate cancer segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train', 'valid' or 'test'.
        sequence: The MRI sequence to use as input. Either 't2', 'adc' or 'dwi'.
        label_type: The type of annotations. Either 'anatomy' (prostate zones) or 'tumor' (cancer lesions).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_prostate158_dataset(
        path, patch_shape, split, sequence, label_type, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
