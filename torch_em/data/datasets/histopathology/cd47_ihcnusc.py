"""The CD47-IHCNuSC dataset contains nucleus annotations for CD47 immunohistochemistry (IHC) images
of esophageal cancer.

The dataset holds 86 patches of 1024 x 1024 pixels at 40x magnification (0.25 micrometer per pixel),
70 for training and 16 for validation / testing, with 18,483 annotated nuclei.
The annotations were made in a human-in-the-loop process: HoverFast created coarse nucleus masks, which were
refined by hand in QuPath over several rounds and then validated by experts. Every nucleus carries one of
seven types (see 'TYPES'). The 'ambiguous' type of the annotation protocol is not part of the release,
so ambiguous nuclei are background in the labels.

The dataset is located at https://doi.org/10.5524/102830 (CC0 1.0).
This dataset is from the publication https://doi.org/10.1093/gigascience/giag091.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal

import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/102000_103000/102830"
ARCHIVES = {
    "CD47-IHCNuSC_images.zip": "016e2f8f0e6efcb8d55ecf9fb22eeffcefd28ef4127e804e6dffe27ce12a21cb",
    "CD47-IHCNuSC_annotations.zip": "90c4e2c799a48a80ceaaa1e0a03d9632ca47b8f82bd7e464d49a4654a879d5e9",
}
TYPES = {
    1: "positive_tumor", 2: "positive_immune", 3: "positive_others", 4: "negative_tumor",
    5: "negative_immune", 6: "negative_others", 7: "negative_stroma",
}


def _create_h5_files(path):
    import h5py
    from scipy.io import loadmat

    for split in ("train", "val"):
        out_dir = os.path.join(path, "preprocessed", split)
        os.makedirs(out_dir, exist_ok=True)
        image_paths = natsorted(glob(os.path.join(path, "images", split, "*.tif")))
        for image_path in tqdm(image_paths, desc=f"Preprocess CD47-IHCNuSC {split} images"):
            name = os.path.splitext(os.path.basename(image_path))[0]
            image = imageio.imread(image_path)[..., :3]
            labels = loadmat(os.path.join(path, "annotations", split, f"{name}.mat"))
            instances, types = labels["inst_map"].astype("uint32"), labels["type_map"].astype("uint8")
            assert image.shape[:2] == instances.shape == types.shape, name

            with h5py.File(os.path.join(out_dir, f"{name}.h5"), "w") as f:
                f.create_dataset("raw", data=image.transpose(2, 0, 1), compression="gzip")
                f.create_dataset("labels/instances", data=instances, compression="gzip")
                f.create_dataset("labels/semantic", data=types, compression="gzip")


def get_cd47_ihcnusc_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CD47-IHCNuSC dataset.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the preprocessed data is stored.
    """
    data_dir = os.path.join(path, "preprocessed")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    for name, checksum in ARCHIVES.items():
        zip_path = os.path.join(path, name)
        util.download_source(zip_path, f"{BASE_URL}/{name}", download, checksum=checksum)
        util.unzip(zip_path, path, remove=False)

    _create_h5_files(path)
    return data_dir


def get_cd47_ihcnusc_paths(
    path: Union[os.PathLike, str], split: Literal["train", "val"], download: bool = False
) -> List[str]:
    """Get paths to the CD47-IHCNuSC data.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        split: The choice of data split. Either 'train' or 'val' (the validation / test split of the authors).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    assert split in ("train", "val"), split
    data_dir = get_cd47_ihcnusc_data(path, download)
    return natsorted(glob(os.path.join(data_dir, split, "*.h5")))


def get_cd47_ihcnusc_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"],
    label_choice: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CD47-IHCNuSC dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'val' (the validation / test split of the authors).
        label_choice: The choice of labels, 'instances' or 'semantic' (the nucleus types, see 'TYPES').
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert label_choice in ("instances", "semantic"), label_choice
    paths = get_cd47_ihcnusc_paths(path, split, download)
    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key="raw",
        label_paths=paths,
        label_key=f"labels/{label_choice}",
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs
    )


def get_cd47_ihcnusc_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val"],
    label_choice: Literal["instances", "semantic"] = "instances",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CD47-IHCNuSC dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split. Either 'train' or 'val' (the validation / test split of the authors).
        label_choice: The choice of labels, 'instances' or 'semantic' (the nucleus types, see 'TYPES').
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cd47_ihcnusc_dataset(path, patch_shape, split, label_choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
