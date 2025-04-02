"""The BriFiSeg dataset contains annotations for nuclei segmentation in brightfield images.

The dataset is located at https://zenodo.org/records/7195636.
This dataset is from the publication https://doi.org/10.48550/arXiv.2211.03072.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

from skimage.measure import label as connected_components

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/7195636/files/1channel.tar"
CHECKSUM = "2be173c265ab737957dd6c007cc51a5ad528087d23cecc53b211cf4dcc7727fa"


def _preprocess_data(path, data_dir):
    import h5py
    import nibabel as nib

    raw_paths = natsorted(glob(os.path.join(path, "1channel", "Task*", "images*", "*.nii.gz")))
    label_paths = natsorted(glob(os.path.join(path, "1channel", "Task*", "labels*", "*.nii.gz")))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    for rpath, lpath in tqdm(zip(raw_paths, label_paths), total=len(raw_paths), desc="Preprocess inputs"):
        raw = nib.load(rpath).get_fdata().squeeze(-1)
        labels = nib.load(lpath).get_fdata().squeeze(-1)
        labels = (labels > 0).astype("uint32")  # binarise all nuclei
        labels = connected_components(labels).astype(labels.dtype)  # running connected components

        fsplit = lpath.split("/")
        fname = fsplit[-1].split(".")[0]
        split = "train" if fsplit[-2] == "labelsTr" else "test"

        os.makedirs(os.path.join(data_dir, split), exist_ok=True)
        with h5py.File(os.path.join(data_dir, split, f"{fname}.h5"), "w") as f:
            f.create_dataset("raw", data=raw)
            f.create_dataset("labels", data=labels)


def get_brifiseg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the BriFiSeg data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    tar_path = os.path.join(path, "1channel.tar")
    util.download_source(path=tar_path, url=URL, checksum=CHECKSUM, download=download)
    util.unzip_tarfile(tar_path=tar_path, dst=path)

    for zip_path in glob(os.path.join(os.path.join(path, "1channel"), "*.zip")):
        util.unzip(zip_path=zip_path, dst=os.path.join(path, "1channel"))

    _preprocess_data(path, data_dir)

    return data_dir


def get_brifiseg_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'test'],
    cell_type: Optional[Literal['A549', 'HELA', 'MCF7', 'RPE1']] = None,
    download: bool = False
) -> List[str]:
    """Get the BriFiSeg data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split.
        cell_type: The choice of cell type.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the input data.
    """
    data_dir = get_brifiseg_data(path, download)

    if split not in ['train', 'test']:
        raise ValueError(f"'{split}' is not a valid split.")

    if cell_type is None:
        cell_type = "*"

    input_paths = natsorted(glob(os.path.join(data_dir, split, f"{cell_type}_*.h5")))
    return input_paths


def get_brifiseg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'],
    cell_type: Optional[Literal['A549', 'HELA', 'MCF7', 'RPE1']] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BriFiSeg dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        cell_type: The choice of cell type.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    input_paths = get_brifiseg_paths(path, split, cell_type, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=input_paths,
        raw_key="raw",
        label_paths=input_paths,
        label_key="labels",
        patch_shape=patch_shape,
        **kwargs
    )


def get_brifiseg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'],
    cell_type: Optional[Literal['A549', 'HELA', 'MCF7', 'RPE1']] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BriFiSeg dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        cell_type: The choice of cell type.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_brifiseg_dataset(path, patch_shape, split, cell_type, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
