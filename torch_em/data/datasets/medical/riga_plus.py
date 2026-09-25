"""The RIGA+ dataset contains annotations for optic disc and optic cup segmentation in fundus images,
gathered from the original RIGA dataset (BinRushed, Magrabia and three MESSIDOR subsets), for the task
of glaucoma assessment and unsupervised domain adaptation.

The dataset is hosted on Zenodo at https://zenodo.org/records/6325549 ("RIGA+ Dataset for Unsupervised
Domain Adaptation in Medical Image Segmentation").

NOTE: There is a second, differently-scoped dataset that also carries the "RIGA+" name, hosted at
https://zenodo.org/records/8009107 ("A Fundus Image Dataset for Domain Generalization in Joint
Segmentation of Optic Disc and Optic Cup"). It combines images from REFUGE, Drishti-GS, ORIGA and RIGA
into a single collection and is not covered by this module.

The five domains provided by this dataset (BinRushed, Magrabia, MESSIDOR_Base1, MESSIDOR_Base2 and
MESSIDOR_Base3) each ship six independent optic disc / cup annotations per image (one per rater). The
label masks are grayscale images with 3 pixel values: 0 (optic cup), 128 (optic disc, excluding the
cup) and 255 (background).

Please cite the dataset if you use it for your research.
"""

import os
from typing import Union, Tuple, Literal, List

import pandas as pd

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/6325549/files/RIGAPlus.zip"
CHECKSUM = "f9fa96df2faa70852ccd4c73d0e79ed6eb258b5a37e72292f5efa3f81f195d20"

DOMAINS = ["BinRushed", "Magrabia", "MESSIDOR_Base1", "MESSIDOR_Base2", "MESSIDOR_Base3"]


def get_riga_plus_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the RIGA+ dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "RIGA")
    if os.path.exists(data_dir):
        return path

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "RIGAPlus.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return path


def get_riga_plus_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "test"],
    domain: Union[str, List[str]] = DOMAINS,
    rater: int = 1,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the RIGA+ data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split, as provided by the official 'train' / 'test' csv files.
        domain: The choice of domain(s) to use. One or several of 'BinRushed', 'Magrabia',
            'MESSIDOR_Base1', 'MESSIDOR_Base2' and 'MESSIDOR_Base3'.
        rater: The choice of rater (1 to 6) for the ground-truth optic disc / cup masks.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    root_dir = get_riga_plus_data(path=path, download=download)

    assert split in ["train", "test"], f"'{split}' is not a valid split."
    assert 1 <= rater <= 6, f"'{rater}' is not a valid rater choice, must be in [1, 6]."

    domains = [domain] if isinstance(domain, str) else domain
    for d in domains:
        assert d in DOMAINS, f"'{d}' is not a valid domain, must be one of {DOMAINS}."

    image_paths, gt_paths = [], []
    for d in domains:
        csv_path = os.path.join(root_dir, f"{d}_{split}.csv")
        df = pd.read_csv(csv_path)
        for image_rel_path, mask_rel_path in zip(df["image"], df["mask"]):
            stem, ext = os.path.splitext(mask_rel_path)
            image_paths.append(os.path.join(root_dir, image_rel_path))
            gt_paths.append(os.path.join(root_dir, f"{stem}-{rater}{ext}"))

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0
    for image_path, gt_path in zip(image_paths, gt_paths):
        assert os.path.exists(image_path), image_path
        assert os.path.exists(gt_path), gt_path

    return image_paths, gt_paths


def get_riga_plus_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    domain: Union[str, List[str]] = DOMAINS,
    rater: int = 1,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the RIGA+ dataset for segmentation of optic disc and optic cup in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        domain: The choice of domain(s) to use.
        rater: The choice of rater for the ground-truth optic disc / cup masks.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_riga_plus_paths(path, split, domain, rater, download)

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


def get_riga_plus_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    domain: Union[str, List[str]] = DOMAINS,
    rater: int = 1,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the RIGA+ dataloader for segmentation of optic disc and optic cup in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        domain: The choice of domain(s) to use.
        rater: The choice of rater for the ground-truth optic disc / cup masks.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_riga_plus_dataset(path, patch_shape, split, domain, rater, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
