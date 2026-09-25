"""The TEVG dataset contains annotations for microvascular segmentation in H&E stained
histology images of tissue-engineered vascular grafts (TEVGs) explanted from sheep carotid
arteries. Each patch is labeled with a semantic mask of 9 histological classes: arteriole
lumen, arteriole media, arteriole adventitia, venule lumen, venule wall, capillary lumen,
capillary wall, immune cells, and nerve trunks.

The dataset is located at https://doi.org/10.5281/zenodo.10838384 under the CC-BY-4.0 license.
This dataset is from the publication https://doi.org/10.3389/fbioe.2024.1411680.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    1: "https://zenodo.org/records/10838384/files/fold_1.zip",
    2: "https://zenodo.org/records/10838384/files/fold_2.zip",
    3: "https://zenodo.org/records/10838384/files/fold_3.zip",
    4: "https://zenodo.org/records/10838384/files/fold_4.zip",
    5: "https://zenodo.org/records/10838384/files/fold_5.zip",
}
CHECKSUMS = {
    1: "96a8be9ed361d2658670e5bde36e406a415ef5a9c30df39442265dbdb0e667a0",
    2: None,
    3: None,
    4: None,
    5: None,
}


def get_tevg_data(path: Union[os.PathLike, str], fold: Literal[1, 2, 3, 4, 5], download: bool = False) -> str:
    """Download the TEVG dataset for one cross-validation fold.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        fold: The choice of cross-validation fold.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder where the fold data is stored.
    """
    fold_dir = os.path.join(path, f"fold_{fold}")
    if os.path.exists(fold_dir):
        return fold_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, f"fold_{fold}.zip")
    util.download_source(path=zip_path, url=URLS[fold], download=download, checksum=CHECKSUMS[fold])
    util.unzip(zip_path=zip_path, dst=path)

    return fold_dir


def get_tevg_paths(
    path: Union[os.PathLike, str],
    fold: Literal[1, 2, 3, 4, 5],
    split: Literal["train", "test"],
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the TEVG data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        fold: The choice of cross-validation fold.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    fold_dir = get_tevg_data(path, fold, download)

    raw_paths = natsorted(glob(os.path.join(fold_dir, split, "img", "*.jpg")))
    label_paths = natsorted(glob(os.path.join(fold_dir, split, "mask", "*.png")))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    assert all(
        os.path.splitext(os.path.basename(raw_path))[0] == os.path.splitext(os.path.basename(label_path))[0]
        for raw_path, label_path in zip(raw_paths, label_paths)
    )

    return raw_paths, label_paths


def get_tevg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    fold: Literal[1, 2, 3, 4, 5] = 1,
    split: Literal["train", "test"] = "train",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the TEVG dataset for microvascular segmentation in tissue-engineered vascular grafts.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        fold: The choice of cross-validation fold.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_tevg_paths(path, fold, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        ndim=2,
        with_channels=True,
        **kwargs,
    )


def get_tevg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    fold: Literal[1, 2, 3, 4, 5] = 1,
    split: Literal["train", "test"] = "train",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the TEVG dataloader for microvascular segmentation in tissue-engineered vascular grafts.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        fold: The choice of cross-validation fold.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_tevg_dataset(path, patch_shape, fold, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
