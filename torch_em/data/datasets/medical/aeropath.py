"""The AeroPath dataset contains annotations for airway and lung segmentation in thoracic CT
(computed tomography angiography) scans of lung cancer patients with challenging pathologies.

The dataset comprises 27 CT volumes with binary annotations for the airways and for the lungs.

NOTE: The label legend is as follows:
- background: 0, airways: 1 (for 'label_choice' = 'airways')
- background: 0, lungs: 1 (for 'label_choice' = 'lungs')

The dataset is located at https://zenodo.org/records/10069289 (see also https://github.com/raidionics/AeroPath).

This dataset is from the publication https://doi.org/10.48550/arXiv.2311.01138.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/10069289/files/AeroPath.zip?download=1"
CHECKSUM = "996b6bd7c79b71a871568293bf6927a52e8e73c1a4dadc1b0975ce3eec3e42ee"


def get_aeropath_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the AeroPath dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    data_dir = os.path.join(path, "AeroPath")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "AeroPath.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_aeropath_paths(
    path: Union[os.PathLike, str], label_choice: Literal["airways", "lungs"] = "airways", download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the AeroPath data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        label_choice: The choice of annotated structure. Either 'airways' or 'lungs'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if label_choice not in ["airways", "lungs"]:
        raise ValueError(f"'{label_choice}' is not a valid label choice. Please choose from 'airways' or 'lungs'.")

    data_dir = get_aeropath_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "*", "*_CT_HR.nii.gz")))
    label_paths = [p.replace("_CT_HR.nii.gz", f"_CT_HR_label_{label_choice}.nii.gz") for p in raw_paths]
    assert len(raw_paths) > 0 and all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_aeropath_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    label_choice: Literal["airways", "lungs"] = "airways",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the AeroPath dataset for airway (or lung) segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of annotated structure. Either 'airways' or 'lungs'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_aeropath_paths(path, label_choice, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_aeropath_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    label_choice: Literal["airways", "lungs"] = "airways",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the AeroPath dataloader for airway (or lung) segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of annotated structure. Either 'airways' or 'lungs'.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_aeropath_dataset(path, patch_shape, label_choice, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
