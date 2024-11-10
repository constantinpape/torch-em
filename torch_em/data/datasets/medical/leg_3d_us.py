"""The LEG 3D US dataset contains annotations for leg muscle segmentation
in 3d ultrasound scans.

NOTE: The label legends are described as follows:
- background: 0
- soleus (SOL): 100
- gastrocnemius medialis (GM): 150
- gastrocnemuis lateralist (GL): 200

The dataset is located at https://www.cs.cit.tum.de/camp/publications/leg-3d-us-dataset/.

This dataset is from the article: https://doi.org/10.1371/journal.pone.0268550.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "train": "https://www.campar.in.tum.de/public_datasets/2024_IPCAI_Vanessa/leg_train_data.zip",
    "val": "https://www.campar.in.tum.de/public_datasets/2024_IPCAI_Vanessa/leg_validation_data.zip",
    "test": "https://www.campar.in.tum.de/public_datasets/2024_IPCAI_Vanessa/leg_test_data.zip",
}

CHECKSUMS = {
    "train": "747e9ada7135979218d93022ac46d40a3a85119e2ea7aebcda4b13f7dfda70d6",
    "val": "c204fa0759dd279de722a423401da60657bc0d1ab5f57d135cd0ad55c32af70f",
    "test": "42ad341e8133f827d35f9cb3afde3ffbe5ae97dc2af448b6f9af6d4ea6ac99f0",
}


def get_leg_3d_us_data(
    path: Union[os.PathLike, str], split: Literal['train', 'val', 'test'], download: bool = False
):
    """Download the LEG 3D US data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.
    """
    data_dir = os.path.join(path, split)
    if os.path.exists(data_dir):
        return

    os.makedirs(path, exist_ok=True)

    if split not in URLS:
        raise ValueError(f"'{split}' is not a valid split choice.")

    zip_name = "validation" if split == "val" else split
    zip_path = os.path.join(path, f"leg_{zip_name}_data.zip")
    util.download_source(path=zip_path, url=URLS[split], download=download, checksum=CHECKSUMS[split])
    util.unzip(zip_path=zip_path, dst=path)


def _preprocess_labels(label_paths):
    neu_label_paths = []
    for lpath in tqdm(label_paths, desc="Preprocessing labels"):
        neu_label_path = lpath.replace(".mha", "_preprocessed.mha")
        neu_label_paths.append(neu_label_path)
        if os.path.exists(neu_label_path):
            continue

        import SimpleITK as sitk

        labels = sitk.ReadImage(lpath)
        larray = sitk.GetArrayFromImage(labels)

        for i, lid in enumerate([100, 150, 200], start=1):
            larray[larray == lid] = i

        sitk_label = sitk.GetImageFromArray(larray)
        sitk.WriteImage(sitk_label, neu_label_path)

    return neu_label_paths


def get_leg_3d_us_paths(
    path: Union[os.PathLike, str], split: Literal['train', 'val', 'test'], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the LEG 3D US data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_leg_3d_us_data(path, split, download)

    raw_paths = natsorted(glob(os.path.join(path, split, "*", "x*.mha")))
    label_paths = [fpath.replace("x", "masksX") for fpath in raw_paths]
    label_paths = _preprocess_labels(label_paths)

    assert len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def get_leg_3d_us_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the LEG 3D US dataset for leg muscle segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        resize_inputs:  Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_leg_3d_us_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_leg_3d_us_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    split: Literal['train', 'val', 'test'],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the LEG 3D US dataloader for leg muscle segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', 'val' or 'test'.
        resize_inputs:  Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_leg_3d_us_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
