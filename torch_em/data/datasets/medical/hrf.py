"""The HRF dataset contains annotations for retinal vessel segmentation in high-resolution
fundus images of healthy, diabetic retinopathy and glaucomatous eyes.

This dataset is located at https://www5.cs.fau.de/research/data/fundus-images/.
The dataset is from the publication https://doi.org/10.1155/2013/154860.
The dataset is licensed under CC BY 4.0 (see https://www5.cs.fau.de/research/data/fundus-images/
for details). Please cite the publication above if you use this dataset for your research.
"""

import os
from glob import glob
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/"

CATEGORIES = {
    "healthy": "healthy",
    "diabetic_retinopathy": "diabetic_retinopathy",
    "glaucoma": "glaucoma",
}

CHECKSUMS = {
    "healthy": "a4ce863b87371cecca8c841c4f7d8b06e39c5298fc78b103c35781bdb7eff389",
    "diabetic_retinopathy": "278630bea52c2096dbfa4b6486647ea60e56db1d67f1308dab2ff4383481a244",
    "glaucoma": "c3428a5eb971ce3165f21463c3e6bf0758531e30035303236b785e6b2d005071",
    "healthy_manualsegm": "2f8c67ee83e9ba16707119aa9ad98ad978c70566d467fe84b328854528ef16b8",
    "diabetic_retinopathy_manualsegm": "5d949ed31d4b825f33b7b14123d21f2562588c8bcbaf33984ddd51ed706d6dbd",
    "glaucoma_manualsegm": "ad3a0a39d2226a66da2c4072e031e23be9d27c8804e9a314db3ae757fc6f910a",
}


def get_hrf_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the HRF dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    os.makedirs(path, exist_ok=True)

    for category in CATEGORIES:
        image_dir = os.path.join(path, category)
        if not os.path.exists(image_dir):
            zip_path = os.path.join(path, f"{category}.zip")
            util.download_source(
                path=zip_path, url=f"{BASE_URL}{category}.zip", download=download, checksum=CHECKSUMS[category],
            )
            util.unzip(zip_path=zip_path, dst=image_dir)

        label_dir = os.path.join(path, f"{category}_manualsegm")
        if not os.path.exists(label_dir):
            zip_path = os.path.join(path, f"{category}_manualsegm.zip")
            util.download_source(
                path=zip_path,
                url=f"{BASE_URL}{category}_manualsegm.zip",
                download=download,
                checksum=CHECKSUMS[f"{category}_manualsegm"],
            )
            util.unzip(zip_path=zip_path, dst=label_dir)

    return path


def get_hrf_paths(
    path: Union[os.PathLike, str],
    category: Literal["healthy", "diabetic_retinopathy", "glaucoma", "all"] = "all",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the HRF data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        category: The choice of eye category. One of 'healthy', 'diabetic_retinopathy', 'glaucoma' or
            'all' (uses all three categories).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if category == "all":
        categories = list(CATEGORIES)
    elif category in CATEGORIES:
        categories = [category]
    else:
        raise ValueError(f"'{category}' is not a valid category. Choose from {list(CATEGORIES) + ['all']}.")

    data_dir = get_hrf_data(path=path, download=download)

    image_paths, label_paths = [], []
    for cat in categories:
        cat_image_paths = sorted(
            glob(os.path.join(data_dir, cat, "*.jpg")) + glob(os.path.join(data_dir, cat, "*.JPG"))
        )
        for image_path in cat_image_paths:
            fname = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(data_dir, f"{cat}_manualsegm", f"{fname}.tif")
            assert os.path.exists(label_path), f"The label at '{label_path}' does not exist."
            image_paths.append(image_path)
            label_paths.append(label_path)

    assert len(image_paths) == len(label_paths) and len(image_paths) > 0

    return image_paths, label_paths


def get_hrf_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    category: Literal["healthy", "diabetic_retinopathy", "glaucoma", "all"] = "all",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HRF dataset for segmentation of retinal blood vessels in high-resolution fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        category: The choice of eye category.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_hrf_paths(path=path, category=category, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_hrf_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    category: Literal["healthy", "diabetic_retinopathy", "glaucoma", "all"] = "all",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the HRF dataloader for segmentation of retinal blood vessels in high-resolution fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        category: The choice of eye category.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_hrf_dataset(path, patch_shape, category, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
