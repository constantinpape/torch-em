"""The IGNITE dataset contains semantic tissue segmentations in H&E-stained NSCLC
and centroid annotations in IHC-stained NSCLC

The dataset is located at https://doi.org/10.5281/zenodo.15674784.
This dataset is from the publication https://doi.org/10.48550/arXiv.2507.16855.
Please cite it if you use this dataset in your research.
"""

import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import pandas as pd
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset

import torch_em

from .. import util

URLS = {
    "tissue_annotations": "https://zenodo.org/records/15674785/files/annotations.zip",
    "images": "https://zenodo.org/records/15674785/files/images.zip",
    "data_overview": "https://zenodo.org/records/15674785/files/data_overview.csv",
}

CHECKSUMS = {
    "tissue_annotations": "b333fab032735de87563c5510de38fc5e2dccc0903a787f7b2b9bd249e66713b",
    "images": "12389313f7f05a6dfb1a15b4aa94a8b16ec4a61a9daf2e86ca6e0a19db2b7628",
    "data_overview": "fa693185d602b9fa91b5556fb622c82c1761759829d593923537f2e774cf8def",
}


def get_split_samples(path: Path, split: str):
    df = pd.read_csv(
        path / "data_overview.csv", index_col="image_path", compression="gzip"
    )
    split_paths = df[(df["split"] == split) & (df["stain"] == "H&E")].index.tolist()
    return [Path(p).name for p in split_paths]


def get_ignite_data(
    path: Path,
    download: bool = False,
    annotation_type: str = "tissue_annotations",
) -> str:
    """Download the IGNITE dataset for tissue segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the downloaded data.
    """

    if annotation_type != "tissue_annotations":
        raise NotImplementedError(
            f"Annotation loading for {annotation_type} is not implemented."
        )

    for data_entity in [annotation_type, "images"]:
        data_dir = path / "data" / data_entity
        if data_dir.exists():
            continue

        data_dir.mkdir(parents=True, exist_ok=True)

        zip_path = path / f"{data_entity}.zip"
        util.download_source(
            path=zip_path,
            url=URLS[data_entity],
            download=download,
            checksum=CHECKSUMS[data_entity],
        )

        util.unzip(zip_path=zip_path, dst=data_dir)

    util.download_source(
        path=data_dir.parent.parent / "data_overview.csv",
        url=URLS["data_overview"],
        download=download,
        checksum=CHECKSUMS["data_overview"],
    )

    return data_dir.parent


def get_ignite_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "test"]] = None,
    annotation_type: Optional[Literal["tissue_annotations"]] = "tissue_annotations",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the LyNSec data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        choice: The choice of dataset.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the image data.
        List of filepaths to the label data.
    """
    data_dir = get_ignite_data(Path(path), download, annotation_type=annotation_type)

    annotation_dir = data_dir / "tissue_annotations" / "annotations" / "he"
    img_dir = data_dir / "images" / "images" / "he"

    if split is not None:
        split_filenames = get_split_samples(path, split)
        img_paths = natsorted([str(img_dir / fn) for fn in split_filenames])
        annotation_paths = natsorted(
            [str(annotation_dir / fn) for fn in split_filenames]
        )
    else:
        img_paths = natsorted(
            [str(p) for p in img_dir.iterdir() if not p.stem.endswith("context")]
        )
        annotation_paths = natsorted(
            [str(p) for p in annotation_dir.iterdir() if not p.stem.endswith("context")]
        )

    return img_paths, annotation_paths


def get_ignite_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "test"]] = None,
    annotation_type: Optional[Literal["tissue_annotations"]] = "tissue_annotations",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the LyNSeC dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        choice: The choice of dataset.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_ignite_paths(path, split, annotation_type, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs,
            patch_shape=patch_shape,
            resize_inputs=resize_inputs,
            resize_kwargs=resize_kwargs,
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs,
    )


def get_ignite_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Optional[Literal["train", "test"]] = None,
    annotation_type: Optional[Literal["tissue_annotations"]] = "tissue_annotations",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the LyNSeC dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        choice: The choice of dataset.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_ignite_dataset(
        path, patch_shape, split, annotation_type, resize_inputs, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
