"""The Cardiomyocyte TEM dataset contains annotations for semantic segmentation of cells, myofibres,
mitochondria, nuclei and nucleoli in transmission electron microscopy (TEM) images of isolated cardiomyocytes.

There are nine large 2D micrographs (3500x magnification) with binary masks. Nucleoli are only annotated in one image.

The dataset is from the publication "Comprehensive Transmission Electron Microscopy Platform for Spatially Resolved
Mitochondrial and Nuclear Architecture in Isolated Cardiomyocytes" and is available at
https://doi.org/10.5281/zenodo.17068504. Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import List, Tuple, Union

from natsort import natsorted
from torch.utils.data import Dataset, DataLoader

import torch_em
from torch_em.transform.label import labels_to_binary

from .. import util


BASE_URL = "https://zenodo.org/records/17068504/files"
RAW_URLS = {
    "914.11_grid1_nucA_m2_M3500x": "61bad379ba42bcc0c8429448dab346811aefb25ff76d07ddf9670a960b99b07c",
    "914.11_grid4_nucA_m2_M3500x": "ee40d1a849df3e1efea0d36d152d63779179c587a03dafc9a44cdcb687eb881e",
    "914.16_18.12.2023_nucD_m2_3500x": "e21fec9443e08f0e8bb49e264622f2f89227fbad4f69126eb076199f9606252e",
    "914.1_nucA_m3_3500x": "eec1368185ce12825281af4a7dbe080332336957b94965af336035418fc32d96",
    "914.1_nucC_m3_3500x": "b4f676d18881b7a451d9a5ecc5f8ff3cafd913105a1bb43dcdd25af07a0aea8f",
    "914.22_nucC_m3_3500x": "a3370477636f69d707f2a79388b5a725ac5a87594ff7b171ba3b76fdd4669a93",
    "914.22_nucE_m2_3500x": "0b2ae97f1b60f1a910958c1478a8f427cdc17d5ef8ab65d0b32e52e689f79d38",
    "914.4_nucA_m3_M3500x": "2b0be4a9f3161be1a96c7ff29b6c3eaa42610fae3ff3c71c68d8e862bb109197",
    "914.9_grid1_nucA_m3_M3500x": "679ffd9fa4eaa619e4fe199bc84b8667d5ca53ef1a9f1c90cf0bbb967fe6e7d6",
}
# NOTE: The mask archives contain more images than the released raw data. Only the ones with raw data are used.
# Some of the masks have a 'KO_' prefix in their name, which is missing in the name of the corresponding raw image.
MASK_ZIPS = {
    "cell": ("Cell_Masks", "dff94c42d1150ffd78657429ba5695c7a2ed91522f5f224bd4ad1366f07ad796"),
    "myo": ("Myo_Masks", "c7e63c6421d0a2f5ecc129d930ab40c7df4c5768f142102a19c86ee674d8342e"),
    "mito": ("Mito_Masks", "50d43cb088ca0e3650881ee10139f0889197d60584994e18c29c95513537eecd"),
    "nucleus": ("Nucleus_Masks", "62ab8a2f1ad72bb2e5766ab00cdd48a74bdb71b6327761527d909551c35857e1"),
    "nucleolus": ("Nucleolus_Masks", "c52455cab0c8c35a219ad8823904a98ca041a6c2725550df059afd82c864d1c1"),
}
LABELS = list(MASK_ZIPS.keys())


def get_cardiomyocyte_tem_data(
    path: Union[os.PathLike, str], label_choice: str = "mito", download: bool = False
) -> str:
    """Download the Cardiomyocyte TEM dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        label_choice: The structure to segment. One of 'cell', 'myo', 'mito', 'nucleus' or 'nucleolus'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the data folder.
    """
    if label_choice not in LABELS:
        raise ValueError(f"'{label_choice}' is not a valid label choice. Please choose from {LABELS}.")

    image_dir = os.path.join(path, "images")
    os.makedirs(image_dir, exist_ok=True)
    for name, checksum in RAW_URLS.items():
        util.download_source(
            os.path.join(image_dir, f"{name}.tif"), f"{BASE_URL}/{name}.tif?download=1", download, checksum
        )

    folder, checksum = MASK_ZIPS[label_choice]
    if not os.path.exists(os.path.join(path, folder)):
        zip_path = os.path.join(path, f"{folder}.zip")
        util.download_source(zip_path, f"{BASE_URL}/{folder}.zip?download=1", download, checksum)
        util.unzip(zip_path, path)

    return path


def get_cardiomyocyte_tem_paths(
    path: Union[os.PathLike, str], label_choice: str = "mito", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the Cardiomyocyte TEM data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        label_choice: The structure to segment. One of 'cell', 'myo', 'mito', 'nucleus' or 'nucleolus'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_cardiomyocyte_tem_data(path, label_choice, download)

    folder, _ = MASK_ZIPS[label_choice]
    mask_paths = glob(os.path.join(path, folder, "*.tif"))

    raw_paths, label_paths = [], []
    for name in RAW_URLS:
        names = (f"{name}_{label_choice}.tif", f"KO_{name}_{label_choice}.tif")
        match = [p for p in mask_paths if os.path.basename(p) in names]
        # Not all images have masks for every structure (e.g. nucleoli).
        if not match:
            continue
        raw_paths.append(os.path.join(path, "images", f"{name}.tif"))
        label_paths.append(match[0])

    assert raw_paths and len(raw_paths) == len(label_paths)
    return natsorted(raw_paths), natsorted(label_paths, key=lambda p: os.path.basename(p).replace("KO_", ""))


def get_cardiomyocyte_tem_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    label_choice: str = "mito",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Cardiomyocyte TEM dataset for cell and organelle segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        label_choice: The structure to segment. One of 'cell', 'myo', 'mito', 'nucleus' or 'nucleolus'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cardiomyocyte_tem_paths(path, label_choice, download)

    kwargs = util.update_kwargs(kwargs, "label_transform", labels_to_binary)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_cardiomyocyte_tem_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    label_choice: str = "mito",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Cardiomyocyte TEM dataloader for cell and organelle segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        label_choice: The structure to segment. One of 'cell', 'myo', 'mito', 'nucleus' or 'nucleolus'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cardiomyocyte_tem_dataset(path, patch_shape, label_choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
