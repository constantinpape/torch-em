"""The TotalSegmentator effusion dataset contains annotations for pleural and pericardial effusion in CT scans.

This is the training dataset for the "pleural_pericard_effusion" task of the TotalSegmentator repository
(https://github.com/wasserth/TotalSegmentator), which is distributed separately from the main
TotalSegmentator dataset (see `torch_em.data.datasets.medical.totalsegmentator`). It consists of 600
CT volumes with a semantic label volume per case, following the class order of the "pleural_pericard_effusion"
task: 0 = background, 1 = lung_pleural, 2 = pleural_effusion, 3 = pericardial_effusion (see `LABEL_IDS`).

The dataset is located at https://doi.org/10.5281/zenodo.20272295 and licensed under CC BY 4.0.

This dataset is part of the TotalSegmentator project, published at https://doi.org/10.1148/ryai.230024.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/20272295/files/Dataset315_pleural_pericard_effusion.zip"
CHECKSUM = "1c8f043b3e466b49af80fa87df6b048b45dcd675796e79a8f3cc2abc18f3bba7"

LABEL_IDS = {"background": 0, "lung_pleural": 1, "pleural_effusion": 2, "pericardial_effusion": 3}


def get_totalsegmentator_effusion_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the TotalSegmentator pleural / pericardial effusion dataset.

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
    zip_path = os.path.join(path, "Dataset315_pleural_pericard_effusion.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def get_totalsegmentator_effusion_paths(
    path: Union[os.PathLike, str], download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the TotalSegmentator effusion data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_totalsegmentator_effusion_data(path, download)

    raw_paths, label_paths = [], []
    for raw_path in sorted(glob(os.path.join(data_dir, "imagesTr", "*_0000.nii.gz"))):
        case_id = os.path.basename(raw_path)[:-len("_0000.nii.gz")]
        label_path = os.path.join(data_dir, "labelsTr", f"{case_id}.nii.gz")
        assert os.path.exists(label_path), label_path
        raw_paths.append(raw_path)
        label_paths.append(label_path)

    assert len(raw_paths) > 0
    return raw_paths, label_paths


def get_totalsegmentator_effusion_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the TotalSegmentator effusion dataset for pleural / pericardial effusion segmentation in CT.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_totalsegmentator_effusion_paths(path, download)

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


def get_totalsegmentator_effusion_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the TotalSegmentator effusion dataloader for pleural / pericardial effusion segmentation in CT.

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
    dataset = get_totalsegmentator_effusion_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
