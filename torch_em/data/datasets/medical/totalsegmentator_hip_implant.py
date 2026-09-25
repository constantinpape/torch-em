"""The TotalSegmentator hip implant dataset contains annotations for hip implants in CT scans.

This is the training dataset for the "hip_implant" task of the TotalSegmentator repository
(https://github.com/wasserth/TotalSegmentator), which is distributed separately from the main
TotalSegmentator dataset (see `torch_em.data.datasets.medical.totalsegmentator`). It consists of 71
CT volumes with a single binary label for hip implants (0 = background, 1 = implant). A small number
of volumes are negative controls with an entirely empty (all-background) label; these are filtered out
by `get_totalsegmentator_hip_implant_paths`.

The dataset is located at https://doi.org/10.5281/zenodo.20272031 and licensed under CC BY 4.0.

This dataset is part of the TotalSegmentator project, published at https://doi.org/10.1148/ryai.230024.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/20272031/files/Dataset260_hip_implant.zip"
CHECKSUM = "c5f7d80ca569f2afb4fe0125dce5b74e9cb14e0ae15da9a2526724d1400aec67"

LABEL_IDS = {"background": 0, "implant": 1}


def get_totalsegmentator_hip_implant_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the TotalSegmentator hip implant dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the 'imagesTr' and 'labelsTr' folders.
    """
    # The archive has no top-level folder, hence it is extracted directly into 'path'.
    data_dir = path
    if os.path.exists(os.path.join(data_dir, "dataset.json")):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "Dataset260_hip_implant.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def get_totalsegmentator_hip_implant_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the TotalSegmentator hip implant data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    import nibabel as nib
    import numpy as np

    data_dir = get_totalsegmentator_hip_implant_data(path, download)

    raw_paths, label_paths = [], []
    for raw_path in sorted(glob(os.path.join(data_dir, "imagesTr", "*_0000.nii.gz"))):
        case_id = os.path.basename(raw_path)[:-len("_0000.nii.gz")]
        label_path = os.path.join(data_dir, "labelsTr", f"{case_id}.nii.gz")
        assert os.path.exists(label_path), label_path

        # Skip the rare negative control case(s), whose label volume is entirely background.
        if not np.any(nib.load(label_path).get_fdata()):
            continue

        raw_paths.append(raw_path)
        label_paths.append(label_path)

    assert len(raw_paths) > 0
    return raw_paths, label_paths


def get_totalsegmentator_hip_implant_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the TotalSegmentator hip implant dataset for hip implant segmentation in CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_totalsegmentator_hip_implant_paths(path, download)

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


def get_totalsegmentator_hip_implant_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the TotalSegmentator hip implant dataloader for hip implant segmentation in CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_totalsegmentator_hip_implant_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
