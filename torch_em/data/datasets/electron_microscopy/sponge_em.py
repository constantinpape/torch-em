"""This dataset contains volume EM data of a sponge chamber with
segmentation annotations for cells, cilia and microvilli.

It contains three annotated volumes. The dataset is part of the publication
https://doi.org/10.1126/science.abj2949. Please cite this publication of you use the
dataset in your research.
"""

import os
from glob import glob
from typing import Optional, Sequence, Tuple, Union

import torch_em
from torch.utils.data import Dataset, DataLoader
from .. import util

URL = "https://zenodo.org/record/8150818/files/sponge_em_train_data.zip?download=1"
CHECKSUM = "f1df616cd60f81b91d7642933e9edd74dc6c486b2e546186a7c1e54c67dd32a5"


def get_sponge_em_data(path: Union[os.PathLike, str], download: bool) -> Tuple[str, int]:
    """Download the SpongeEM training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The path to the downloaded data.
        The number of downloaded volumes.
    """
    n_files = len(glob(os.path.join(path, "*.h5")))
    if n_files == 3:
        return path, n_files
    elif n_files == 0:
        pass
    else:
        raise RuntimeError(
            f"Invalid number of downloaded files in {path}. Please remove this folder and rerun this function."
        )

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "data.zip")
    util.download_source(zip_path, URL, download, CHECKSUM)
    util.unzip(zip_path, path)

    n_files = len(glob(os.path.join(path, "*.h5")))
    assert n_files == 3
    return path, n_files


def get_sponge_em_dataset(
    path: Union[os.PathLike, str],
    mode: str,
    patch_shape: Tuple[int, int, int],
    sample_ids: Optional[Sequence[int]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the SpongeEM dataset for the segmentation of structures in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        mode: Choose the segmentation task, either 'semantic' or 'instances'.
        patch_shape: The patch shape to use for training.
        sample_ids: The sample to download, valid ids are 1, 2 and 3.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """

    assert mode in ("semantic", "instances")
    data_folder, n_files = get_sponge_em_data(path, download)

    if sample_ids is None:
        sample_ids = range(1, n_files + 1)
    paths = [os.path.join(data_folder, f"train_data_0{i}.h5") for i in sample_ids]

    raw_key = "volumes/raw"
    label_key = f"volumes/labels/{mode}"
    return torch_em.default_segmentation_dataset(paths, raw_key, paths, label_key, patch_shape, **kwargs)


def get_sponge_em_loader(
    path: Union[os.PathLike, str],
    mode: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    sample_ids: Optional[Sequence[int]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the SpongeEM dataloader for the segmentation of structures in EM.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        mode: Choose the segmentation task, either 'semantic' or 'instances'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        sample_ids: The sample to download, valid ids are 1, 2 and 3.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
       The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_sponge_em_dataset(path, mode, patch_shape, sample_ids=sample_ids, download=download, **ds_kwargs)
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
