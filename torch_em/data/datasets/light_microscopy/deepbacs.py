"""DeepBacs is a dataset for segmenting bacteria in label-free light microscopy.

This dataset is from the publication https://doi.org/10.1038/s42003-022-03634-z.
Please cite it if you use this dataset in your research.
"""

import os
import shutil
from glob import glob
from typing import Tuple, Union

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util

URLS = {
    "s_aureus": "https://zenodo.org/record/5550933/files/DeepBacs_Data_Segmentation_Staph_Aureus_dataset.zip?download=1",  # noqa
    "e_coli": "https://zenodo.org/record/5550935/files/DeepBacs_Data_Segmentation_E.coli_Brightfield_dataset.zip?download=1",  # noqa
    "b_subtilis": "https://zenodo.org/record/5639253/files/Multilabel_U-Net_dataset_B.subtilis.zip?download=1",
    "mixed": "https://zenodo.org/record/5551009/files/DeepBacs_Data_Segmentation_StarDist_MIXED_dataset.zip?download=1",
}
CHECKSUMS = {
    "s_aureus": "4047792f1248ee82fce34121d0ade84828e55db5a34656cc25beec46eacaf307",
    "e_coli": "f812a2f814c3875c78fcc1609a2e9b34c916c7a9911abbf8117f423536ef1c17",
    "b_subtilis": "1",
    "mixed": "2730e6b391637d6dc05bbc7b8c915fd8184d835ac3611e13f23ac6f10f86c2a0",
}


def _assort_val_set(path, bac_type):
    image_paths = glob(os.path.join(path, bac_type, "training", "source", "*"))
    image_paths = [os.path.split(_path)[-1] for _path in image_paths]

    val_partition = 0.2
    # let's get a balanced set of bacterias, if bac_type is mixed
    if bac_type == "mixed":
        _sort_1, _sort_2, _sort_3 = [], [], []
        for _path in image_paths:
            if _path.startswith("JE2"):
                _sort_1.append(_path)
            elif _path.startswith("pos"):
                _sort_2.append(_path)
            elif _path.startswith("train_"):
                _sort_3.append(_path)

        val_image_paths = [
            *np.random.choice(_sort_1, size=int(val_partition * len(_sort_1)), replace=False),
            *np.random.choice(_sort_2, size=int(val_partition * len(_sort_2)), replace=False),
            *np.random.choice(_sort_3, size=int(val_partition * len(_sort_3)), replace=False)
        ]
    else:
        val_image_paths = np.random.choice(image_paths, size=int(val_partition * len(image_paths)), replace=False)

    val_image_dir = os.path.join(path, bac_type, "val", "source")
    val_label_dir = os.path.join(path, bac_type, "val", "target")
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    for sample_id in val_image_paths:
        src_val_image_path = os.path.join(path, bac_type, "training", "source", sample_id)
        dst_val_image_path = os.path.join(val_image_dir, sample_id)
        shutil.move(src_val_image_path, dst_val_image_path)

        src_val_label_path = os.path.join(path, bac_type, "training", "target", sample_id)
        dst_val_label_path = os.path.join(val_label_dir, sample_id)
        shutil.move(src_val_label_path, dst_val_label_path)


def get_deepbacs_data(path: Union[os.PathLike, str], bac_type: str, download: bool) -> str:
    f"""Download the DeepBacs training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        bac_type: The bacteria type. The available types are:
            {', '.join(URLS.keys())}
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the training data.
    """
    bac_types = list(URLS.keys())
    assert bac_type in bac_types, f"{bac_type} is not in expected bacteria types: {bac_types}"

    data_folder = os.path.join(path, bac_type)
    if os.path.exists(data_folder):
        return data_folder

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"{bac_type}.zip")
    if not os.path.exists(zip_path):
        util.download_source(zip_path, URLS[bac_type], download, checksum=CHECKSUMS[bac_type])
    util.unzip(zip_path, os.path.join(path, bac_type))

    # Get a val split for the expected bacteria type.
    _assort_val_set(path, bac_type)
    return data_folder


def get_deepbacs_paths(
    path: Union[os.PathLike, str], bac_type: str, split: str, download: bool = False
) -> Tuple[str, str]:
    f"""Get paths to the DeepBacs data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use for the dataset. Either 'train', 'val' or 'test'.
        bac_type: The bacteria type. The available types are:
            {', '.join(URLS.keys())}
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder where image data is stored.
        Filepath to the folder where label data is stored.
    """
    get_deepbacs_data(path, bac_type, download)

    # the bacteria types other than mixed are a bit more complicated so we don't have the dataloaders for them yet
    # mixed is the combination of all other types
    if split == "train":
        dir_choice = "training"
    else:
        dir_choice = split

    if bac_type != "mixed":
        raise NotImplementedError(f"Currently only the bacteria type 'mixed' is supported, not {bac_type}")

    image_folder = os.path.join(path, bac_type, dir_choice, "source")
    label_folder = os.path.join(path, bac_type, dir_choice, "target")

    return image_folder, label_folder


def get_deepbacs_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    bac_type: str = "mixed",
    download: bool = False,
    **kwargs
) -> Dataset:
    f"""Get the DeepBacs dataset for bacteria segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use for the dataset. Either 'train', 'val' or 'test'.
        patch_shape: The patch shape to use for training.
        bac_type: The bacteria type. The available types are:
            {', '.join(URLS.keys())}
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    assert split in ("train", "val", "test")

    image_folder, label_folder = get_deepbacs_paths(path, bac_type, split, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_folder,
        raw_key="*.tif",
        label_paths=label_folder,
        label_key="*.tif",
        patch_shape=patch_shape,
        **kwargs
    )


def get_deepbacs_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    bac_type: str = "mixed",
    download: bool = False,
    **kwargs
) -> DataLoader:
    f"""Get the DeepBacs dataset for bacteria segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to use for the dataset. Either 'train', 'val' or 'test'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        bac_type: The bacteria type. The available types are:
            {', '.join(URLS.keys())}
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_deepbacs_dataset(path, split, patch_shape, bac_type=bac_type, download=download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
