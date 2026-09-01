"""The TNBC-CellType dataset contains annotations for nucleus segmentation and cell type
classification in H&E stained histopathology images. It extends the original TNBC breast
cancer dataset with 18 additional brain section images from TCGA.

The dataset is located at https://doi.org/10.5281/zenodo.3552674 under the CC-BY-4.0 license.
This dataset is from the publication https://arxiv.org/abs/2207.10950.
Please cite it if you use this dataset for your research.
"""

import os
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import json
import pandas as pd
import imageio.v3 as imageio
from scipy.ndimage import binary_fill_holes
from sklearn.model_selection import train_test_split
from bioimage_cpp.segmentation import label as connected_components

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "raw": "https://zenodo.org/records/3552674/files/TNBC_and_Brain_dataset.zip",
    "celltype": "https://zenodo.org/records/3552674/files/TNBC_and_Brain_dataset_celltype_integer.zip",
}
CHECKSUMS = {
    "raw": "fa8a71748e04f5e6b8c9b5cdb79bb44c0096b6c0fd899c4b963e2e16fc521cf3",
    "celltype": "1f4aa20b7384e7893cea05626eb22ed0a6d67dd525ff8129069cdacde4114963",
}


def _create_split_csv(path, data_dir, split):
    csv_path = os.path.join(path, 'tnbc_celltype_split.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df[split] = df[split].apply(lambda x: json.loads(x.replace("'", '"')))  # ensures all items from column in list.
        split_list = df.iloc[0][split]

    else:
        print(f"Creating a new split file at '{csv_path}'.")
        image_names = [
            os.path.basename(image).split(".")[0] for image in glob(os.path.join(data_dir, '*.h5'))
        ]

        train_ids, test_ids = train_test_split(image_names, test_size=0.2)  # 20% for test split.
        train_ids, val_ids = train_test_split(train_ids, test_size=0.15)  # 15% for val split.
        split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

        df = pd.DataFrame.from_dict([split_ids])
        df.to_csv(csv_path, index=False)

        split_list = split_ids[split]

    return split_list


def _preprocess_images(path):
    import h5py

    raw_paths = natsorted(glob(os.path.join(path, "TNBC_and_Brain_dataset", "Slide_*", "*.png")))
    celltype_paths = natsorted(
        glob(os.path.join(path, "TNBC_and_Brain_dataset_celltype_integer", "GT_*", "*.png"))
    )
    assert len(raw_paths) == len(celltype_paths) and len(raw_paths) > 0

    preprocessed_dir = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)

    for rpath, cpath in tqdm(zip(raw_paths, celltype_paths), desc="Preprocessing images", total=len(raw_paths)):
        raw = imageio.imread(rpath)
        if raw.ndim == 3 and raw.shape[-1] == 4:
            raw = raw[..., :-1]  # remove 4th alpha channel (seems like an empty channel).

        raw = raw.transpose(2, 0, 1)
        celltype = imageio.imread(cpath)
        # A handful of the binary traces have an isolated 1px hole from annotation rounding;
        # filling it never changes the instance count, so this is safe everywhere.
        instances = connected_components(binary_fill_holes(celltype > 0))

        vol_path = os.path.join(preprocessed_dir, f"{Path(cpath).stem}.h5")

        with h5py.File(vol_path, "w") as f:
            f.create_dataset("raw", shape=raw.shape, data=raw, compression="gzip")
            f.create_dataset("labels/semantic", shape=celltype.shape, data=celltype, compression="gzip")
            f.create_dataset("labels/instances", shape=instances.shape, data=instances, compression="gzip")

    shutil.rmtree(os.path.join(path, "TNBC_and_Brain_dataset"))
    shutil.rmtree(os.path.join(path, "TNBC_and_Brain_dataset_celltype_integer"))
    shutil.rmtree(os.path.join(path, "__MACOSX"), ignore_errors=True)


def get_tnbc_celltype_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the TNBC-CellType dataset for nucleus segmentation and cell type classification.

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

    for name, url in URLS.items():
        zip_path = os.path.join(path, f"{name}.zip")
        util.download_source(path=zip_path, url=url, download=download, checksum=CHECKSUMS[name])
        util.unzip(zip_path=zip_path, dst=path)

    _preprocess_images(path)

    return data_dir


def get_tnbc_celltype_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False
) -> List[str]:
    """Get paths to the TNBC-CellType data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the preprocessed image data.
    """
    data_dir = get_tnbc_celltype_data(path, download)
    split_list = _create_split_csv(path, data_dir, split)
    volume_paths = [os.path.join(data_dir, f"{fname}.h5") for fname in split_list]
    return volume_paths


def get_tnbc_celltype_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    label_choice: Literal["semantic", "instances"] = "instances",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the TNBC-CellType dataset for nucleus segmentation and cell type classification.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        label_choice: The choice of label. Either 'instances' for nucleus instance segmentation
            or 'semantic' for cell type classification.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_tnbc_celltype_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

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


def get_tnbc_celltype_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    label_choice: Literal["semantic", "instances"] = "instances",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the TNBC-CellType dataloader for nucleus segmentation and cell type classification.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        label_choice: The choice of label. Either 'instances' for nucleus instance segmentation
            or 'semantic' for cell type classification.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_tnbc_celltype_dataset(path, patch_shape, split, label_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
