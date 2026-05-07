"""This dataset comes from the Neurips Cell Segmentation Challenge,
which collects microscopy images and annotations for cell segmentation.

The dataset contains both images with annotations for cell segmentation
and unlabed images for self-supervised or semi-supervised learning.
See also the challenge website for details: https://neurips22-cellseg.grand-challenge.org/.
The dataset os decribed in the publication https://doi.org/10.1038/s41592-024-02233-6.
Please cite it if you use the dataset in your research.
"""

import os
from glob import glob
from typing import Union, Tuple, Any, Optional, List

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util
from ... import ImageCollectionDataset, RawImageCollectionDataset, ConcatDataset


URL = {
    "train": "https://zenodo.org/records/10719375/files/Training-labeled.zip",
    "val": "https://zenodo.org/records/10719375/files/Tuning.zip",
    "test": "https://zenodo.org/records/10719375/files/Testing.zip",
    "unlabeled": "https://zenodo.org/records/10719375/files/train-unlabeled-part1.zip",
    "unlabeled_wsi": "https://zenodo.org/records/10719375/files/train-unlabeled-part2.zip"
}

CHECKSUM = {
    "train": "b2383929eb8e99b2716fa0d4e2f6e03983e626a57cf00fe85175869c54aa3592",
    "val": "849423d36bb8fcc2d91a5b189a3b6d93c3d4071c9701eaaa44ba393a510459c4",
    "test": "3379730221f43830d30fddf131750e967c9c9bdf04f98811e852a050eb659ccc",
    "unlabeled": "390b38b398b05e9e5306a024a3bd48ab22e49592cfab3c1a119eab3636b38e0d",
    "unlabeled_wsi": "d1e68eba2918305eab8b846e7578ac14683de970e3fa6a7c2a4a55753be56204"
}


DIR_NAMES = {
    "train": "Training-labeled", "val": "Tuning", "test": "Testing/Public",
    "unlabeled": "release-part1", "unlabeled_wsi": "train-unlabeled-part2"
}

ZIP_PATH = {
    "train": "Training-labeled.zip", "val": "Tuning.zip", "test": "Testing.zip",
    "unlabeled": "train-unlabeled-part1.zip", "unlabeled_wsi": "train-unlabeled-part2.zip"
}


def to_rgb(image):
    if image.ndim == 2:
        image = np.concatenate([image[None]] * 3, axis=0)

    if image.ndim == 3 and image.shape[-1] == 3:
        image = image.transpose(2, 0, 1)

    assert image.ndim == 3
    assert image.shape[0] == 3, f"{image.shape}"
    return image


def get_neurips_cellseg_data(root: Union[os.PathLike, str], split: str, download: bool) -> str:
    f"""Download the Neurips Cell Seg training data.

    Args:
        root: Filepath to a folder where the downloaded data will be saved.
        split: The data split to download. Available splits are:
            {', '.join(URL.keys())}
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the training data.
    """
    os.makedirs(root, exist_ok=True)

    target_dir = os.path.join(root, DIR_NAMES[split])
    zip_path = os.path.join(root, ZIP_PATH[split])

    if not os.path.exists(target_dir):
        util.download_source(path=zip_path, url=URL[split], download=download, checksum=CHECKSUM[split])
        util.unzip(zip_path=zip_path, dst=root)

    return target_dir


def get_neurips_cellseg_paths(
    root: Union[os.PathLike, str], split: str, download: bool = False
) -> Tuple[List[str], List[str]]:
    f"""Get paths to NeurIPS CellSeg Challenge data.

    Args:
        root: Filepath to a folder where the downloaded data will be saved.
        split: The data split to download. Available splits are:
            {', '.join(URL.keys())}
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    path = get_neurips_cellseg_data(root, split, download)

    image_folder = os.path.join(path, "images")
    assert os.path.exists(image_folder)
    label_folder = os.path.join(path, "labels")
    assert os.path.exists(label_folder)

    all_image_paths = glob(os.path.join(image_folder, "*"))
    all_image_paths.sort()
    all_label_paths = glob(os.path.join(label_folder, "*"))
    all_label_paths.sort()
    assert len(all_image_paths) == len(all_label_paths)

    return all_image_paths, all_label_paths


def get_neurips_cellseg_supervised_dataset(
    root: Union[str, os.PathLike],
    split: str,
    patch_shape: Tuple[int, int],
    make_rgb: bool = True,
    label_transform: Optional[Any] = None,
    label_transform2: Optional[Any] = None,
    raw_transform: Optional[Any] = None,
    transform: Optional[Any] = None,
    label_dtype: torch.dtype = torch.float32,
    n_samples: Optional[int] = None,
    sampler: Optional[Any] = None,
    download: bool = False,
) -> Dataset:
    f"""Get the dataset for cell segmentation from the NeurIPS Cell Seg Challenge.

    Args:
        root: Filepath to a folder where the downloaded data will be saved.
        split: The data split to download. Available splits are:
            {', '.join(URL.keys())}
        patch_shape: The patch shape to use for training.
        make_rgb: Whether to map all data to RGB or treat it as grayscale.
        label_transform: Transformation of labels, applied before data augmentation.
        label_transform2: Transformation of labels, applied after data augmentation.
        raw_transform: Transformation of the raw data.
        label_dtype: The data type of the label data.
        n_samples: Number of samples per epoch from this dataset.
        sampler: Sampler for rejecting batches.
        download: Whether to download the data if it is not present.

    Returns:
        The segmentation dataset.
    """
    assert split in ("train", "val", "test"), split
    image_paths, label_paths = get_neurips_cellseg_paths(root, split, download)

    if raw_transform is None:
        trafo = to_rgb if make_rgb else None
        raw_transform = torch_em.transform.get_raw_transform(augmentation2=trafo)

    if transform is None:
        transform = torch_em.transform.get_augmentations(ndim=2)

    return ImageCollectionDataset(
        raw_image_paths=image_paths,
        label_image_paths=label_paths,
        patch_shape=patch_shape,
        raw_transform=raw_transform,
        label_transform=label_transform,
        label_transform2=label_transform2,
        label_dtype=label_dtype,
        transform=transform,
        n_samples=n_samples,
        sampler=sampler
    )


def get_neurips_cellseg_supervised_loader(
    root: Union[str, os.PathLike],
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    make_rgb: bool = True,
    label_transform: Optional[Any] = None,
    label_transform2: Optional[Any] = None,
    raw_transform: Optional[Any] = None,
    transform: Optional[Any] = None,
    label_dtype: torch.dtype = torch.float32,
    n_samples: Optional[Any] = None,
    sampler: Optional[Any] = None,
    download: bool = False,
    **loader_kwargs
) -> DataLoader:
    f"""Get the dataset for cell segmentation from the NeurIPS Cell Seg Challenge.

    Args:
        root: Filepath to a folder where the downloaded data will be saved.
        split: The data split to download. Available splits are:
            {', '.join(URL.keys())}
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        make_rgb: Whether to map all data to RGB or treat it as grayscale.
        label_transform: Transformation of labels, applied before data augmentation.
        label_transform2: Transformation of labels, applied after data augmentation.
        raw_transform: Transformation of the raw data.
        transform: Transformation applied to raw and label data.
        label_dtype: The data type of the label data.
        n_samples: Number of samples per epoch from this dataset.
        sampler: Sampler for rejecting batches.
        download: Whether to download the data if it is not present.
        loader_kwargs: Keyword arguments for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds = get_neurips_cellseg_supervised_dataset(
        root=root,
        split=split,
        patch_shape=patch_shape,
        make_rgb=make_rgb,
        label_transform=label_transform,
        label_transform2=label_transform2,
        raw_transform=raw_transform,
        transform=transform,
        label_dtype=label_dtype,
        n_samples=n_samples,
        sampler=sampler,
        download=download
    )
    return torch_em.segmentation.get_data_loader(ds, batch_size, **loader_kwargs)


def _get_image_paths(root, download):
    path = get_neurips_cellseg_data(root, "unlabeled", download)
    image_paths = glob(os.path.join(path, "*"))
    image_paths.sort()
    return image_paths


def _get_wholeslide_paths(root, patch_shape, download):
    path = get_neurips_cellseg_data(root, "unlabeled_wsi", download)
    image_paths = glob(os.path.join(path, "*"))
    image_paths.sort()

    # one of the whole slides doesn't support memmap which will make it very slow to load
    image_paths = [path for path in image_paths if torch_em.util.supports_memmap(path)]
    assert len(image_paths) > 0

    n_samples = 0
    for im_path in image_paths:
        shape = torch_em.util.load_image(im_path).shape
        assert len(shape) == 3 and shape[-1] == 3
        im_shape = shape[:2]
        n_samples += np.prod([sh // psh for sh, psh in zip(im_shape, patch_shape)])

    return image_paths, n_samples


def get_neurips_cellseg_unsupervised_dataset(
    root: Union[str, os.PathLike],
    patch_shape: Tuple[int, int],
    make_rgb: bool = True,
    raw_transform: Optional[Any] = None,
    transform: Optional[Any] = None,
    dtype: torch.dtype = torch.float32,
    sampler: Optional[Any] = None,
    use_images: bool = True,
    use_wholeslide: bool = True,
    download: bool = False,
) -> Dataset:
    """Get the unsupervised dataset from the NeurIPS Cell Seg Challenge.

    Args:
        root: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        make_rgb: Whether to map all data to RGB or treat it as grayscale.
        raw_transform: Transformation of the raw data.
        transform: Transformation applied to raw and label data.
        dtype: The data type of the image data.
        sampler: Sampler for rejecting batches.
        use_images: Whether to use the normal image data.
        use_wholeslide: Whether to use the wholeslide image data.
        download: Whether to download the data if it is not present.

    Returns:
        The segmentation dataset.
    """
    if raw_transform is None:
        trafo = to_rgb if make_rgb else None
        raw_transform = torch_em.transform.get_raw_transform(augmentation2=trafo)
    if transform is None:
        transform = torch_em.transform.get_augmentations(ndim=2)

    datasets = []
    if use_images:
        image_paths = _get_image_paths(root, download)
        datasets.append(
            RawImageCollectionDataset(
                raw_image_paths=image_paths,
                patch_shape=patch_shape,
                raw_transform=raw_transform,
                transform=transform,
                dtype=dtype,
                sampler=sampler
            )
        )
    if use_wholeslide:
        image_paths, n_samples = _get_wholeslide_paths(root, patch_shape, download)
        datasets.append(
            RawImageCollectionDataset(
                raw_image_paths=image_paths,
                patch_shape=patch_shape,
                raw_transform=raw_transform,
                transform=transform,
                dtype=dtype,
                n_samples=n_samples,
                sampler=sampler
            )
        )
    assert len(datasets) > 0
    return ConcatDataset(*datasets)


def get_neurips_cellseg_unsupervised_loader(
    root: Union[str, os.PathLike],
    patch_shape: Tuple[int, int],
    batch_size: int,
    make_rgb: bool = True,
    raw_transform: Optional[Any] = None,
    transform: Optional[Any] = None,
    dtype: torch.dtype = torch.float32,
    sampler: Optional[Any] = None,
    use_images: bool = True,
    use_wholeslide: bool = True,
    download: bool = False,
    **loader_kwargs,
) -> DataLoader:
    """Get the unsupervised dataset from the NeurIPS Cell Seg Challenge.

    Args:
        root: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        make_rgb: Whether to map all data to RGB or treat it as grayscale.
        raw_transform: Transformation of the raw data.
        transform: Transformation applied to raw and label data.
        dtype: The data type of the image data.
        sampler: Sampler for rejecting batches.
        use_images: Whether to use the normal image data.
        use_wholeslide: Whether to use the wholeslide image data.
        download: Whether to download the data if it is not present.
        loader_kwargs: Keyword arguments for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds = get_neurips_cellseg_unsupervised_dataset(
        root=root, patch_shape=patch_shape, make_rgb=make_rgb, raw_transform=raw_transform, transform=transform,
        dtype=dtype, sampler=sampler, use_images=use_images, use_wholeslide=use_wholeslide, download=download
    )
    return torch_em.segmentation.get_data_loader(ds, batch_size, **loader_kwargs)
