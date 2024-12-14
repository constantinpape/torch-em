"""The CPM dataset contains annotations for nucleus segmentation in
H&E stained histopathology images for different tissue images.

NOTE: You must download the files manually.
1. The dataset is located at https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK.
2. The restructuring details are mentioned by the authors here: https://github.com/vqdang/hover_net/issues/5#issuecomment-508431862.  # noqa

This dataset is from the publication https://doi.org/10.3389/fbioe.2019.00053.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Literal, Optional, Tuple, List

from scipy.io import loadmat
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = {
    "cpm15": "https://drive.google.com/drive/folders/11ko-GcDsPpA9GBHuCtl_jNzWQl6qY_-I?usp=drive_link",
    "cpm17": "https://drive.google.com/drive/folders/1sJ4nmkif6j4s2FOGj8j6i_Ye7z9w0TfA?usp=drive_link",
}


def get_cpm_data(
    path: Union[os.PathLike, str], data_choice: Literal['cpm15', 'cpm17'], download: bool = False
) -> str:
    """Obtain the CPM data.

    NOTE: The dataset is located at https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK.
    Visit the drive link -> select the dataset(s) of choice -> right click and 'Download' the folder as zipfile.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        data_choice: The choice of data.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data has been manually downloaded and later preprocessed.
    """
    if data_choice not in ['cpm15', 'cpm17']:
        raise ValueError(f"'{data_choice}' is not a valid data choice.")

    data_dir = os.path.join(path, data_choice)
    if os.path.exists(data_dir):
        return data_dir

    if download:
        raise NotImplementedError(
            "The dataset cannot be automatically downloaded. "
            "Please see 'get_cpm_data' in 'torch_em/data/datasets/histopathology/cpm.py' for details."
        )

    os.makedirs(path, exist_ok=True)
    zip_path = glob(os.path.join(path, f"{data_choice}*.zip"))
    if len(zip_path) == 0:
        raise AssertionError(
            f"zip file for '{data_choice}' dataset is not found. Please download it from '{URL[data_choice]}'."
        )

    zip_path = zip_path[0]
    util.unzip(zip_path=zip_path, dst=path, remove=False)

    return data_dir


def get_cpm_paths(
    path: Union[os.PathLike, str], data_choice: Literal['cpm15', 'cpm17'], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the CPM data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        data_choice: The choice of data.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the image data.
        List of filepaths to the label data.
    """
    data_dir = get_cpm_data(path, data_choice, download)

    if data_choice == "cpm15":
        raw_dir, label_dir = "Images", "Labels"
    else:
        raw_dir, label_dir = "*/Images", "*/Labels"

    raw_paths = [p for p in natsorted(glob(os.path.join(data_dir, raw_dir, "*.png")))]
    label_mat_paths = [p for p in natsorted(glob(os.path.join(data_dir, label_dir, "*.mat")))]

    label_paths = []
    for mpath in tqdm(label_mat_paths, desc="Preprocessing labels"):
        label_path = mpath.replace(".mat", "_instance_labels.tif")
        label_paths.append(label_path)
        if os.path.exists(label_path):
            continue

        label = loadmat(mpath)["inst_map"]
        imageio.imwrite(label_path, label, compression="zlib")

    assert len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def get_cpm_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    data_choice: Optional[Literal['cpm15', 'cpm17']] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CPM dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        data_choice: The choice of data.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cpm_paths(path, data_choice, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_cpm_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    data_choice: Optional[Literal['cpm15', 'cpm17']] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CPM dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        data_choice: The choice of data.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cpm_dataset(path, patch_shape, data_choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
