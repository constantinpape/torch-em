"""The TopBrain dataset contains annotations for whole brain vessel anatomy segmentation
in computed tomography angiography (CTA) and magnetic resonance angiography (MRA).

The data was curated for the TopBrain 2025 challenge (https://topbrain2025.grand-challenge.org), which
extends the TopCoW challenge from the Circle of Willis to over 40 landmark brain vessel anatomies. This
module downloads the batch-1 training data release, which consists of 30 annotated angiographies (15 CTA
and 15 MRA), reusing the same 15 patients that are also part of the TopCoW dataset (`torch_em.data.datasets.
medical.topcow`). The organizers plan to release more data in future batches; this module only exposes the
batch-1 data that is currently public. The modality is selected with the 'modality' argument ('ct' or 'mr').

The multi-class segmentations label the brain vessel anatomy, see the label ids in the 'itksnap_labelmap_txt'
folder of the downloaded data (40 classes for CTA, 42 classes for MRA, both include the 13 Circle of Willis
classes from the TopCoW dataset).

The data is located at https://doi.org/10.5281/zenodo.16623496.

This dataset is the successor of the TopCoW challenge, published as https://doi.org/10.1056/aidbp2500994.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/16623496/files/TopBrain_Data_Release_Batch1_073025.zip"
CHECKSUM = "468af6ba9a3ff36c9a61c4e02dbb737d304219aec0e3b1f77df5d97e36304a35"

MODALITIES = ["ct", "mr"]


def get_topbrain_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the TopBrain dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "TopBrain_Data_Release_Batch1_073025")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "TopBrain_Data_Release_Batch1_073025.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_topbrain_paths(
    path: Union[os.PathLike, str],
    modality: Optional[Literal["ct", "mr"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the TopBrain data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The angiography modality. Either 'ct' or 'mr'. If None, both modalities are returned.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_topbrain_data(path, download)

    if modality is not None and modality not in MODALITIES:
        raise ValueError(f"'{modality}' is not a valid modality. Please choose one of {MODALITIES}.")

    modalities = MODALITIES if modality is None else [modality]

    raw_paths, label_paths = [], []
    for mod in modalities:
        this_label_paths = natsorted(glob(os.path.join(data_dir, f"labelsTr_topbrain_{mod}", "*.nii.gz")))
        # The images carry the channel suffix '_0000' of the nnU-Net format, the labels do not.
        this_raw_paths = [
            os.path.join(data_dir, f"imagesTr_topbrain_{mod}", os.path.basename(p).replace(".nii.gz", "_0000.nii.gz"))
            for p in this_label_paths
        ]
        assert len(this_raw_paths) > 0 and all(os.path.exists(p) for p in this_raw_paths)
        raw_paths.extend(this_raw_paths)
        label_paths.extend(this_label_paths)

    return raw_paths, label_paths


def get_topbrain_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    modality: Optional[Literal["ct", "mr"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the TopBrain dataset for whole brain vessel anatomy segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        modality: The angiography modality. Either 'ct' or 'mr'. If None, both modalities are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_topbrain_paths(path, modality, download)

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


def get_topbrain_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    modality: Optional[Literal["ct", "mr"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the TopBrain dataloader for whole brain vessel anatomy segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        modality: The angiography modality. Either 'ct' or 'mr'. If None, both modalities are returned.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_topbrain_dataset(path, patch_shape, modality, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
