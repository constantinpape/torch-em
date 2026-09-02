"""The DCSA-Net dataset contains binary nucleus segmentation masks for H&E stained
prostate cancer histopathology images from two internal cohorts (RUMC, YUHS).

NOTE: The masks are binary nucleus foreground/background maps, not per-nucleus instance
labels. Visual inspection also suggests the annotations are not exhaustive: some visible
nuclei in the raw images have no corresponding mask region.

This dataset is located at https://doi.org/10.6084/m9.figshare.22249291.
This dataset is from the publication https://doi.org/10.3389/fonc.2023.1009681.
Please cite it if you use this dataset for your research.
"""

import os
from natsort import natsorted
from typing import Union, Literal, Tuple, List

import json
import pandas as pd
import imageio.v3 as imageio
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/39539971"
CHECKSUM = "25bb4a37672809c7f762a20929218cef838e94c1c316ad1e9c31d801447197ad"


def get_dcsa_net_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the DCSA-Net data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the dataset is downloaded and stored for further preprocessing.
    """
    data_dir = os.path.join(path, "Training Data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "dcsa_net.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _raw_mask_pairs(data_dir):
    image_dir = os.path.join(data_dir, "Train_Images")
    mask_dir = os.path.join(data_dir, "Train_Masks")

    # The archive stores the RUMC cohort as '.bmp' (indices 1-45) and the YUHS cohort as
    # '.jpg' (indices 1-30) under a shared 'Train_Images' folder, while all 75 masks live
    # in a single flat 'Train_Masks' sequence: RUMC masks 1-45, then YUHS masks 46-75.
    pairs = []
    for index in range(1, 46):
        pairs.append((
            os.path.join(image_dir, f"Prostate ({index}).bmp"),
            os.path.join(mask_dir, f"Prostate ({index}).bmp"),
        ))
    for index in range(1, 31):
        pairs.append((
            os.path.join(image_dir, f"Prostate ({index}).jpg"),
            os.path.join(mask_dir, f"Prostate ({index + 45}).bmp"),
        ))

    for raw_path, mask_path in pairs:
        assert os.path.exists(raw_path), raw_path
        assert os.path.exists(mask_path), mask_path

    return pairs


def _convert_masks_to_binary(data_dir, pairs):
    converted_dir = os.path.join(data_dir, "Train_Masks_binary")
    os.makedirs(converted_dir, exist_ok=True)

    label_paths = []
    for _, mask_path in pairs:
        out_path = os.path.join(converted_dir, os.path.basename(mask_path))
        if not os.path.exists(out_path):
            mask = imageio.imread(mask_path)
            if mask.ndim == 3:
                mask = mask[..., 0]
            imageio.imwrite(out_path, (mask > 127).astype("uint8"))
        label_paths.append(out_path)

    return label_paths


def _create_split_csv(path, raw_paths):
    csv_path = os.path.join(path, "dcsa_net_split.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return {split: json.loads(df.iloc[0][split].replace("'", '"')) for split in ("train", "val", "test")}

    print(f"Creating a new split file at '{csv_path}'.")
    image_ids = natsorted(os.path.basename(p) for p in raw_paths)

    train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.15, random_state=42)
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

    df = pd.DataFrame.from_dict([split_ids])
    df.to_csv(csv_path, index=False)

    return split_ids


def get_dcsa_net_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the DCSA-Net data.

    NOTE: The source publishes no official split, so this function creates and stores a
    deterministic split (65% train, 15% val, 20% test) the first time it is called.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the image data.
        List of filepaths to the label data.
    """
    data_dir = get_dcsa_net_data(path, download)
    pairs = _raw_mask_pairs(data_dir)
    converted_label_paths = _convert_masks_to_binary(data_dir, pairs)
    raw_paths = [raw_path for raw_path, _ in pairs]

    split_ids = _create_split_csv(path, raw_paths)[split]
    kept = natsorted(
        (raw_path, label_path) for raw_path, label_path in zip(raw_paths, converted_label_paths)
        if os.path.basename(raw_path) in split_ids
    )
    raw_paths = [raw_path for raw_path, _ in kept]
    label_paths = [label_path for _, label_path in kept]

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    return raw_paths, label_paths


def get_dcsa_net_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DCSA-Net dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_dcsa_net_paths(path, split, download)

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
        is_seg_dataset=False,
        patch_shape=patch_shape,
        with_channels=True,
        ndim=2,
        **kwargs
    )


def get_dcsa_net_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DCSA-Net dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_dcsa_net_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
