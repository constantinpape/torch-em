"""The SKM-TEA dataset contains annotations for knee tissue segmentation in quantitative double-echo
steady-state (qDESS) knee MRI.

It comprises 155 scans with two echoes each and dense segmentations of six soft tissue classes. The scans
come with an official train / val / test split. The label ids are:
- 0: background
- 1: patellar cartilage
- 2: femoral cartilage
- 3: tibial cartilage (medial)
- 4: tibial cartilage (lateral)
- 5: meniscus (medial)
- 6: meniscus (lateral)

NOTE: The label ids were verified against the official dataset documentation
(https://github.com/StanfordMIMI/skm-tea/blob/main/DATASET.md), which documents the order of the one-hot
encoded 'seg' volumes, and on the data: the six channels do not overlap and the by far largest one is the
femoral cartilage.

NOTE: The full dataset with all 155 scans requires registration and cannot be downloaded automatically.
Only the official sample of 3 scans (one per split), which the authors published at
https://huggingface.co/datasets/arjundd/skm-tea-mini, is downloaded if `download` is set to True.
Note that this sample must not be used for reporting metrics. To use the full dataset, follow these steps:
- Visit https://aimi.stanford.edu/skm-tea-knee-mri and follow the link to the dataset on the Stanford AIMI
  shared datasets portal (https://stanfordaimi.azurewebsites.net/datasets/4aaeafb9-c6e6-4e3c-9188-3aaaf0e0a9e7).
- Log in or create an account and follow the download instructions (the full dataset is ~900 GB compressed,
  but only the 'DICOM' track ('image_files') and the annotations are required here).
- Extract the data such that '<path>/image_files/<scan_id>.h5' and
  '<path>/annotations/<version>/{train,val,test}.json' exist, i.e. '<path>' is the 'skm-tea' folder described
  in https://github.com/StanfordMIMI/skm-tea/blob/main/DATASET.md.

The dataset is located at https://aimi.stanford.edu/skm-tea-knee-mri.

This dataset is from the publication https://openreview.net/forum?id=YDMFgD_qJuA.
Please cite it if you use this dataset in your research.
"""

import os
import json
import warnings
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


SAMPLE_URL = "https://huggingface.co/datasets/arjundd/skm-tea-mini/resolve/main/v1-release"

SAMPLE_CHECKSUMS = {
    "image_files.tar.gz": "18af509e31f4468dbb5b237bf1a124508d00ade4ace0b90fae62947ce4709812",
    "train.json": "7a2b80f6ecae2eabe2489235302ac0003ac39f1266c4d356abbf5279f0ccdb7a",
    "val.json": "5d64909b69f89a101755ace309c4769c5884758957efa35d4da27b3b4b455986",
    "test.json": "b057cf984d7fcdd52318642ea2ec3a9241464575c3a91dc29f64e4fef51dad74",
}

SAMPLE_VERSION = "v1.0.0"

LABEL_IDS = {
    "background": 0,
    "patellar_cartilage": 1,
    "femoral_cartilage": 2,
    "tibial_cartilage_medial": 3,
    "tibial_cartilage_lateral": 4,
    "meniscus_medial": 5,
    "meniscus_lateral": 6,
}


def _preprocess_inputs(path, image_dir):
    import h5py

    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    for scan_path in tqdm(natsorted(glob(os.path.join(image_dir, "*.h5"))), desc="Preprocessing the SKM-TEA scans"):
        volume_path = os.path.join(preprocessed_dir, os.path.basename(scan_path))
        if os.path.exists(volume_path):
            continue

        with h5py.File(scan_path, "r") as f:
            echo1, echo2, seg = f["echo1"][:], f["echo2"][:], f["seg"][:]

        # The one-hot encoded segmentation (X, Y, Z, 6) is converted to a label volume with the ids 1 to 6.
        labels = np.where(seg.any(axis=-1), np.argmax(seg, axis=-1) + 1, 0).astype(np.uint8)

        # The volumes are stored as (X, Y, Z) with Z (LR) being the sagittal slice direction.
        # We move the slice axis to the front, so that 2d training samples sagittal slices.
        # The file is written to a temporary path first, so that an interrupted run does not leave a corrupt file.
        with h5py.File(f"{volume_path}.tmp", "w") as f:
            f.create_dataset("raw/echo1", data=echo1.transpose(2, 0, 1), compression="gzip")
            f.create_dataset("raw/echo2", data=echo2.transpose(2, 0, 1), compression="gzip")
            rss = np.sqrt(echo1.astype("float32") ** 2 + echo2.astype("float32") ** 2)
            f.create_dataset("raw/rss", data=rss.transpose(2, 0, 1), compression="gzip")
            f.create_dataset("labels", data=labels.transpose(2, 0, 1), compression="gzip")

        os.rename(f"{volume_path}.tmp", volume_path)

    return preprocessed_dir


def _download_sample_data(path, download):
    msg = "Only the official sample of 3 SKM-TEA scans is downloaded, not the full dataset with 155 scans. "
    msg += "See 'torch_em.data.datasets.medical.skm_tea' for how to obtain the full dataset."
    warnings.warn(msg)

    os.makedirs(path, exist_ok=True)

    tar_path = os.path.join(path, "image_files.tar.gz")
    util.download_source(
        path=tar_path,
        url=f"{SAMPLE_URL}/tarball/image_files.tar.gz",
        download=download,
        checksum=SAMPLE_CHECKSUMS["image_files.tar.gz"],
    )
    util.unzip_tarfile(tar_path=tar_path, dst=path)

    annotation_dir = os.path.join(path, "annotations", SAMPLE_VERSION)
    os.makedirs(annotation_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        split_path = os.path.join(annotation_dir, f"{split}.json")
        if os.path.exists(split_path):
            continue

        util.download_source(
            path=split_path,
            url=f"{SAMPLE_URL}/annotations/{SAMPLE_VERSION}/{split}.json",
            download=download,
            checksum=SAMPLE_CHECKSUMS[f"{split}.json"],
        )


def get_skm_tea_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the SKM-TEA dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is preprocessed.
    """
    preprocessed_dir = os.path.join(path, "preprocessed")
    if os.path.exists(preprocessed_dir) and len(glob(os.path.join(preprocessed_dir, "*.h5"))) > 0:
        return preprocessed_dir

    image_dir = os.path.join(path, "image_files")
    if not os.path.exists(image_dir):
        if not download:
            raise FileNotFoundError(
                f"It's expected to place the downloaded SKM-TEA 'image_files' folder at '{image_dir}'. "
                "See 'torch_em.data.datasets.medical.skm_tea' for the manual download instructions."
            )

        _download_sample_data(path, download)

    return _preprocess_inputs(path, image_dir)


def get_skm_tea_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    version: str = "v1.0.0",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the SKM-TEA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        version: The version of the annotations that define the split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_skm_tea_data(path, download)

    if split not in ["train", "val", "test"]:
        raise ValueError(f"'{split}' is not a valid split.")

    split_path = os.path.join(path, "annotations", version, f"{split}.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Could not find the annotation file for the '{split}' split at '{split_path}'.")

    with open(split_path, "r") as f:
        scan_ids = [image["scan_id"] for image in json.load(f)["images"]]

    volume_paths = natsorted([os.path.join(data_dir, f"{scan_id}.h5") for scan_id in scan_ids])
    volume_paths = [p for p in volume_paths if os.path.exists(p)]
    assert len(volume_paths) > 0, f"Could not find any preprocessed scans for the '{split}' split."

    return volume_paths, volume_paths


def get_skm_tea_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    echo: Literal["echo1", "echo2", "rss"] = "echo1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SKM-TEA dataset for knee tissue segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        echo: The qDESS echo to use as input. Either 'echo1', 'echo2' or 'rss' (root-sum-of-squares of both).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if echo not in ["echo1", "echo2", "rss"]:
        raise ValueError(f"'{echo}' is not a valid echo. Choose one of 'echo1', 'echo2' or 'rss'.")

    raw_paths, label_paths = get_skm_tea_paths(path, split, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=f"raw/{echo}",
        label_paths=label_paths,
        label_key="labels",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_skm_tea_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal["train", "val", "test"],
    echo: Literal["echo1", "echo2", "rss"] = "echo1",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SKM-TEA dataloader for knee tissue segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        echo: The qDESS echo to use as input. Either 'echo1', 'echo2' or 'rss' (root-sum-of-squares of both).
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_skm_tea_dataset(path, patch_shape, split, echo, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
