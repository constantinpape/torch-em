"""The RIM-ONE DL dataset contains annotations for optic disc and optic cup segmentation in fundus
images, for the task of glaucoma assessment.

The dataset is hosted at https://github.com/miag-ull/rim-one-dl. It comprises 485 retinographies
(313 from normal subjects and 172 from patients with glaucoma), collected at three Spanish hospitals
(Hospital Universitario de Canarias, Hospital Universitario Miguel Servet and Hospital Clinico
Universitario San Carlos). The images and reference segmentations are distributed via the bit.ly
links referenced in the repository's README (https://bit.ly/rim-one-dl-images and
https://bit.ly/rim-one-dl-reference-segmentations), which resolve to Google Drive files; this module
downloads directly from the resolved Google Drive files.

The dataset ships two official partitions, each with a training and a test set: 'random' (images
distributed randomly between the two sets) and 'hospital' (the test set is built from two of the
three hospitals, held out from the training set). The label masks are binary PNGs, one for the optic
disc and one for the optic cup, produced with the DCSeg annotation tool by an expert in glaucoma.

The dataset is from the publication https://doi.org/10.5566/ias.2346.
Please cite it if you use this dataset for your research. Data included in this database can only be
used for research and educational purposes.
"""

import os
from glob import glob
from pathlib import Path
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = {
    "images": "https://drive.google.com/uc?id=1teYi_smpLiNZNJcTWdxXgKKLW2fkUQr4",
    "segmentations": "https://drive.google.com/uc?id=1eb1V9V65TuwFNYmYsIzgdAgdWyD6o7bG",
}

CHECKSUM = {
    "images": "85aed5f95c794f52d11b6ed953032108a66576ffcff4b587365df467491b603d",
    "segmentations": "edc363b1f0deabc8ed7a356250a9e1fb21825bf06f7337889fd1e764e1947901",
}

PARTITION_DIRS = {"random": "partitioned_randomly", "hospital": "partitioned_by_hospital"}
SPLIT_DIRS = {"train": "training_set", "test": "test_set"}
TASK_NAMES = {"disc": "Disc", "cup": "Cup"}


def get_rim_one_dl_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the RIM-ONE DL dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    images_dir = os.path.join(path, "RIM-ONE_DL_images")
    if os.path.exists(images_dir):
        return path

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "rim_one_dl_images.zip")
    util.download_source_gdrive(
        path=zip_path, url=URL["images"], download=download, checksum=CHECKSUM["images"], download_type="zip",
    )
    util.unzip(zip_path=zip_path, dst=path)

    zip_path = os.path.join(path, "rim_one_dl_segmentations.zip")
    util.download_source_gdrive(
        path=zip_path, url=URL["segmentations"], download=download, checksum=CHECKSUM["segmentations"],
        download_type="zip",
    )
    util.unzip(zip_path=zip_path, dst=path)

    return path


def get_rim_one_dl_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    partition: Literal["random", "hospital"] = "random",
    task: Literal["disc", "cup"] = "disc",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the RIM-ONE DL data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        partition: The choice of the official partition, either 'random' or 'hospital'.
        task: The choice of labels for the specific task.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    root_dir = get_rim_one_dl_data(path=path, download=download)

    assert split in SPLIT_DIRS, f"'{split}' is not a valid split."
    assert partition in PARTITION_DIRS, f"'{partition}' is not a valid partition."
    assert task in TASK_NAMES, f"'{task}' is not a valid task."

    image_dir = os.path.join(root_dir, "RIM-ONE_DL_images", PARTITION_DIRS[partition], SPLIT_DIRS[split])
    image_paths = sorted(glob(os.path.join(image_dir, "*", "*.png")))

    gt_dir = os.path.join(root_dir, "RIM-ONE_DL_reference_segmentations")
    gt_paths = [
        os.path.join(gt_dir, Path(p).parent.name, f"{Path(p).stem}-1-{TASK_NAMES[task]}-T.png") for p in image_paths
    ]

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0
    for gt_path in gt_paths:
        assert os.path.exists(gt_path), gt_path

    return image_paths, gt_paths


def get_rim_one_dl_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    partition: Literal["random", "hospital"] = "random",
    task: Literal["disc", "cup"] = "disc",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the RIM-ONE DL dataset for segmentation of optic disc and optic cup in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        partition: The choice of the official partition.
        task: The choice of labels for the specific task.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_rim_one_dl_paths(path, split, partition, task, download)

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


def get_rim_one_dl_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    partition: Literal["random", "hospital"] = "random",
    task: Literal["disc", "cup"] = "disc",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the RIM-ONE DL dataloader for segmentation of optic disc and optic cup in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        partition: The choice of the official partition.
        task: The choice of labels for the specific task.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_rim_one_dl_dataset(path, patch_shape, split, partition, task, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
