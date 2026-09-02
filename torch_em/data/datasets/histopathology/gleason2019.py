"""The Gleason2019 dataset contains pixel-level Gleason grade annotations for H&E
stained prostate cancer tissue microarray (TMA) cores.

Mask label values: 0 (benign), 1 (Gleason pattern 3), 2 (Gleason pattern 4),
3 (Gleason pattern 5), 4 (unlabelled).

The test split (TMA 80) carries two independent pathologist annotations.

This dataset is located at https://doi.org/10.7910/DVN/OCYCMP.
This dataset is from the publication https://doi.org/10.1038/s41598-018-30535-1.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Literal, Tuple, List

import numpy as np
import imageio.v3 as imageio
from PIL import Image

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


RAW_URLS = {
    "ZT111_4_A": "https://dataverse.harvard.edu/api/access/datafile/3201629",
    "ZT111_4_B": "https://dataverse.harvard.edu/api/access/datafile/3201630",
    "ZT111_4_C": "https://dataverse.harvard.edu/api/access/datafile/3201631",
    "ZT199_1_A": "https://dataverse.harvard.edu/api/access/datafile/3201632",
    "ZT199_1_B": "https://dataverse.harvard.edu/api/access/datafile/3201633",
    "ZT204_6_A": "https://dataverse.harvard.edu/api/access/datafile/3201634",
    "ZT204_6_B": "https://dataverse.harvard.edu/api/access/datafile/3201635",
    "ZT76_39_A": "https://dataverse.harvard.edu/api/access/datafile/3201623",
    "ZT76_39_B": "https://dataverse.harvard.edu/api/access/datafile/3201625",
    "ZT80_38_A": "https://dataverse.harvard.edu/api/access/datafile/3201626",
    "ZT80_38_B": "https://dataverse.harvard.edu/api/access/datafile/3201627",
    "ZT80_38_C": "https://dataverse.harvard.edu/api/access/datafile/3201628",
}

RAW_CHECKSUMS = {
    "ZT111_4_A": "4e87448fe2db959a757c792069df5d49aaf08f305484a829832cb2d44e60fba0",
    "ZT111_4_B": "6e898136490c5fc4d46bb66541519a4bdfe77bcb7fcc7d2f39440195e8fe01ce",
    "ZT111_4_C": "85eeeeefd89f55d4aa12ce498f573c55f2ed83bf957b73a736a446cdb7540b45",
    "ZT199_1_A": "2e1f4cc38097d36be6d33c76688ec85df87c7f53812b205e34c1d5c47a720b45",
    "ZT199_1_B": "fa2d79e4891c5043f4dbb3db92e8d75de8d2f7c9afdf51d48c80dfb4f1d0b507",
    "ZT204_6_A": "ad02fe27bbdae2d2af0bf4e9725d8dabe1c15bdb49125e9e0131705401cdb610",
    "ZT204_6_B": "b89d3b8422cb1c2a9f3034d4f146ba0d3ce82260ab512c5ea5fc983506205c14",
    "ZT76_39_A": "054be24c8522bdacef2916a48e2463da238e39efbb96c4aae6e2f802168cc14f",
    "ZT76_39_B": "9d18c74ea72e6936a7ec6481373191b0c3da4f1a915280439b25efa7bcb645dc",
    "ZT80_38_A": "3ae19e2a3b00c4158171487da7cab10f05cbaddbb256a41ad6be73f4be091972",
    "ZT80_38_B": "406211b0822d165555346ae459db6ddb1daa32c9257f1578aeb7c2da03f3c910",
    "ZT80_38_C": "16bda74826ac8c907c85460b72f713f4f63d765167c18b2729743eed7e806048",
}

MASK_URLS = {
    "train": "https://dataverse.harvard.edu/api/access/datafile/3201636",
    "test_pathologist1": "https://dataverse.harvard.edu/api/access/datafile/3201654",
    "test_pathologist2": "https://dataverse.harvard.edu/api/access/datafile/3201655",
}

MASK_CHECKSUMS = {
    "train": "f4a3ce4a599b210d60cddc6ad3cd8d28d58546fbdb82ee59ba52fc66e90698a6",
    "test_pathologist1": "265c31f163072c7cdab9bcd850748b9196e79ac68497449dad542a3095f388ff",
    "test_pathologist2": "2c996b8479aad8ff442ec0e23c80fec4aa9cc39961751252145058266e1267a2",
}

SPLIT_TMAS = {
    "train": ["ZT111_4_A", "ZT111_4_B", "ZT111_4_C", "ZT199_1_A", "ZT199_1_B", "ZT204_6_A", "ZT204_6_B"],
    "val": ["ZT76_39_A", "ZT76_39_B"],
    "test": ["ZT80_38_A", "ZT80_38_B", "ZT80_38_C"],
}


def get_gleason2019_data(
    path: Union[os.PathLike, str], split: Literal["train", "val", "test"], download: bool = False,
) -> str:
    """Download the Gleason2019 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the dataset is downloaded and stored for further preprocessing.
    """
    os.makedirs(path, exist_ok=True)

    for stem in SPLIT_TMAS[split]:
        if os.path.exists(os.path.join(path, stem)):
            continue
        tar_path = os.path.join(path, f"{stem}.tar.gz")
        util.download_source(path=tar_path, url=RAW_URLS[stem], download=download, checksum=RAW_CHECKSUMS[stem])
        util.unzip_tarfile(tar_path=tar_path, dst=path)

    mask_key = "train" if split in ("train", "val") else "test_pathologist1"
    if not os.path.exists(os.path.join(path, f"Gleason_masks_{mask_key}")):
        tar_path = os.path.join(path, f"Gleason_masks_{mask_key}.tar.gz")
        util.download_source(
            path=tar_path, url=MASK_URLS[mask_key], download=download, checksum=MASK_CHECKSUMS[mask_key]
        )
        util.unzip_tarfile(tar_path=tar_path, dst=path)

    if split == "test" and not os.path.exists(os.path.join(path, "Gleason_masks_test_pathologist2")):
        tar_path = os.path.join(path, "Gleason_masks_test_pathologist2.tar.gz")
        util.download_source(
            path=tar_path, url=MASK_URLS["test_pathologist2"], download=download,
            checksum=MASK_CHECKSUMS["test_pathologist2"],
        )
        util.unzip_tarfile(tar_path=tar_path, dst=path)

    return path


def get_gleason2019_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "test"],
    test_pathologist: Literal[1, 2] = 1,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Gleason2019 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        test_pathologist: The choice of pathologist annotation to use for the 'test' split.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the image data.
        List of filepaths to the label data.
    """
    get_gleason2019_data(path, split, download)

    if split == "test":
        mask_dir = os.path.join(path, f"Gleason_masks_test_pathologist{test_pathologist}")
        mask_prefix = f"mask{test_pathologist}_"
    else:
        mask_dir = os.path.join(path, "Gleason_masks_train")
        mask_prefix = "mask_"

    converted_dir = f"{mask_dir}_indexed"
    os.makedirs(converted_dir, exist_ok=True)

    raw_paths, label_paths = [], []
    for stem in SPLIT_TMAS[split]:
        for raw_path in natsorted(glob(os.path.join(path, stem, "*.jpg"))):
            mask_name = f"{mask_prefix}{os.path.basename(raw_path)[:-len('.jpg')]}.png"
            mask_path = os.path.join(mask_dir, mask_name)
            if not os.path.exists(mask_path):
                continue
            label_path = os.path.join(converted_dir, mask_name)
            if not os.path.exists(label_path):
                # The released masks are palette-indexed PNGs (0-4); keep the raw
                # palette index instead of the RGB colors most readers expand them to.
                mask = Image.open(mask_path)
                assert mask.mode == "P", f"Expected a palette-indexed mask, got mode '{mask.mode}': {mask_path}"
                imageio.imwrite(label_path, np.array(mask).astype("uint8"))
            raw_paths.append(raw_path)
            label_paths.append(label_path)

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0
    return raw_paths, label_paths


def get_gleason2019_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    test_pathologist: Literal[1, 2] = 1,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Gleason2019 dataset for Gleason pattern segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        test_pathologist: The choice of pathologist annotation to use for the 'test' split.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_gleason2019_paths(path, split, test_pathologist, download)

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


def get_gleason2019_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "test"],
    test_pathologist: Literal[1, 2] = 1,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Gleason2019 dataloader for Gleason pattern segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        test_pathologist: The choice of pathologist annotation to use for the 'test' split.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_gleason2019_dataset(path, patch_shape, split, test_pathologist, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
