"""The AutoPET dataset contains annotations for lesion segmentation in whole-body FDG-PET/CT scans.

This dataset is from the `AutoPET II - Automated Lesion Segmentation in PET/CT - Domain Generalization` challenge.
Link: https://autopet-ii.grand-challenge.org/

Please cite it if you use this dataset for publication.
"""

import os
from glob import glob
from typing import Tuple, Optional, Union, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


AUTOPET_DATA = "http://193.196.20.155/data/autoPET/data/nifti.zip"
CHECKSUM = "0ac2186ea6d936ff41ce605c6a9588aeb20f031085589897dbab22fc82a12972"


def get_autopet_data(path: Union[os.PathLike, str], download: bool = False):
    """Download the AutoPET dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    target_dir = os.path.join(path, "AutoPET-II")
    if os.path.exists(target_dir):
        return

    os.makedirs(target_dir)

    zip_path = os.path.join(path, "autopet.zip")
    print("The AutoPET data is not available yet and will be downloaded.")
    print("Note that this dataset is large, so this step can take several hours (depending on your internet).")
    util.download_source(path=zip_path, url=AUTOPET_DATA, download=download, checksum=CHECKSUM)
    util.unzip(zip_path, target_dir, remove=False)


def get_autopet_paths(
    path: Union[os.PathLike, str], modality: Optional[Literal["CT", "PET"]] = None, download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the AutoPET adta.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The choice of imaging modality.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_autopet_data(path, download)

    root_dir = os.path.join(path, "AutoPET-II", "FDG-PET-CT-Lesions", "*", "*")
    ct_paths = sorted(glob(os.path.join(root_dir, "CTres.nii.gz")))
    pet_paths = sorted(glob(os.path.join(root_dir, "SUV.nii.gz")))
    label_paths = sorted(glob(os.path.join(root_dir, "SEG.nii.gz")))

    if modality is None:
        raw_paths = [(ct_path, pet_path) for ct_path, pet_path in zip(ct_paths, pet_paths)]
    else:
        if modality == "CT":
            raw_paths = ct_paths
        elif modality == "PET":
            raw_paths = pet_paths
        else:
            raise ValueError("Choose from the available modalities: `CT` / `PET`")

    return raw_paths, label_paths


def get_autopet_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Optional[Literal["CT", "PET"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AutoPET dataset for lesion segmentation in whole-bod FDG-PET/CT scans.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The choice of imaging modality.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_autopet_paths(path, modality, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        with_channels=modality is None,
        **kwargs
    )

    if "sampler" in kwargs:
        for ds in dataset.datasets:
            ds.max_sampling_attempts = 5000

    return dataset


def get_autopet_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Optional[Literal["CT", "PET"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the AutoPET dataloader for lesion segmentation in whole-bod FDG-PET/CT scans.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The choice of imaging modality.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_autopet_dataset(path, patch_shape, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
