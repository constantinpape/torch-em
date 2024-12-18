"""The TNBC dataset contains annotations for nucleus segmentation
in H&E stained histopathology images.

The dataset is located at https://doi.org/10.5281/zenodo.1175282.
Please cite it if you use this dataset for your research.
"""

import os
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List
import random
import pandas as pd
import imageio.v3 as imageio
from skimage.measure import label as connected_components

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/1175282/files/TNBC_NucleiSegmentation.zip"
CHECKSUM = "da708c3a988f4ad4b9bbb9283b387faf703f0bc0e5e689927306bd27ea13a57f"


def create_split_csv(path): # TNBC
    image_names = [os.path.basename(image).split(".")[0] for image in glob(os.path.join(path, 'preprocessed', '*.h5'))]
    split_index = int(len(image_names)*0.8)
    random.shuffle(image_names)
    train_set = image_names[:split_index]
    test_images = image_names[split_index:] 
    val_split_index = int(len(train_set)*0.8)
    train_images = train_set[:val_split_index]
    val_images = train_set[val_split_index:]
    split_data = []
    for img in natsorted(train_images):
        split_data.append({"image_name": img, "split": "train"})
    for img in natsorted(test_images):
        split_data.append({"image_name": img, "split": "test"})
    for img in natsorted(val_images):
        split_data.append({"image_name": img, "split": "val"})
    output_csv = os.path.join(path, 'tnbc_split.csv')
    df = pd.DataFrame(split_data)
    df.to_csv(output_csv, index=False)


def _preprocess_images(path):
    import h5py

    raw_paths = natsorted(glob(os.path.join(path, "TNBC_NucleiSegmentation", "Slide_*", "*.png")))
    label_paths = natsorted(glob(os.path.join(path, "TNBC_NucleiSegmentation", "GT_*", "*.png")))

    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    for rpath, lpath in tqdm(zip(raw_paths, label_paths), desc="Preprocessing images", total=len(raw_paths)):
        raw = imageio.imread(rpath)
        if raw.ndim == 3 and raw.shape[-1] == 4:
            raw = raw[..., :-1]  # remove 4th alpha channel (seems like an empty channel).

        raw = raw.transpose(2, 0, 1)
        label = imageio.imread(lpath)

        vol_path = os.path.join(preprocessed_dir, f"{Path(lpath).stem}.h5")

        with h5py.File(vol_path, "w") as f:
            f.create_dataset("raw", shape=raw.shape, data=raw, compression="gzip")
            f.create_dataset("labels/semantic", shape=label.shape, data=label, compression="gzip")
            f.create_dataset(
                "labels/instances", shape=label.shape, data=connected_components(label), compression="gzip"
            )

    shutil.rmtree(os.path.join(path, "TNBC_NucleiSegmentation"))
    shutil.rmtree(os.path.join(path, "__MACOSX"))


def get_tnbc_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the TNBC dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the downloaded data.
    """
    data_dir = os.path.join(path, "preprocessed")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "TNBC_NucleiSegmentation.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    _preprocess_images(path)

    return data_dir


def get_tnbc_paths(path: Union[os.PathLike, str], download: bool = False) -> List[int]:
    """Get paths to the TNBC data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the preprocessed image data.
    """
    data_dir = get_tnbc_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    return volume_paths


def get_tnbc_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, int], download: bool = False, **kwargs
) -> Dataset:
    """Get the TNBC dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    label_choice = "instances"  # semantic / instances

    volume_paths = get_tnbc_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=f"labels/{label_choice}",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        with_channels=True,
        **kwargs
    )


def get_tnbc_loader(
    path: Union[os.PathLike, str], batch_size: int, patch_shape: Tuple[int, int], download: bool = False, **kwargs
) -> DataLoader:
    """Get the TNBC dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_tnbc_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
