"""The CTSpine1K dataset contains annotations for 25 vertebrae in CT scans.

The dataset consists of 1005 CT scans that were collected from four public collections and annotated with
the individual vertebrae for this release. The 'source' argument selects one of them: 'colonog' (CT
colonography), 'covid19', 'hnscc' (head and neck) and 'msd_liver' (the liver task of the Medical
Segmentation Decathlon). The label ids are 1 to 7 for the cervical vertebrae C1 to C7, 8 to 19 for the
thoracic vertebrae T1 to T12 and 20 to 25 for the lumbar vertebrae L1 to L6. See also `CLASS_IDS`.

NOTE: A scan only covers a part of the spine, so each one contains a contiguous range of the label ids
rather than all 25.

NOTE: The scans of the CT colonography collection are named after the id of their DICOM series, whose
dots hide the nifti extension from the file readers, so the data is linked under a prepared name.

The dataset is located at https://huggingface.co/datasets/alexanderdann/CTSpine1K and is distributed
under the CC BY-NC-SA license.
This dataset is from the publication https://doi.org/10.48550/arXiv.2105.14711.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Optional, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


REPO_ID = "alexanderdann/CTSpine1K"

SOURCES = {
    "colonog": "COLONOG",
    "covid19": "COVID-19",
    "hnscc": "HNSCC-3DCT-RT",
    "msd_liver": "MSD-T10",
}
"""Mapping from the source choice to its folder in the release."""

CLASS_NAMES = (
    [f"c{i}" for i in range(1, 8)] + [f"t{i}" for i in range(1, 13)] + [f"l{i}" for i in range(1, 7)]
)
"""The vertebrae of the CTSpine1K dataset. The label id of a vertebra is its 1-based index."""

CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
"""Mapping from the vertebra name to its label id."""


def _prepare_name(path, prepared_dir, folder, stem):
    """Link a scan under a name without a dot in its stem, which the file readers need to find its extension."""
    out_path = os.path.join(prepared_dir, folder, f"{stem.replace('.', '_')}.nii.gz")
    if not os.path.exists(out_path):
        os.symlink(os.path.abspath(path), out_path)
    return out_path


def get_ctspine1k_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the CTSpine1K dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "raw_data")
    if os.path.exists(data_dir) and glob(os.path.join(data_dir, "volumes", "*", "*.nii.gz")):
        return data_dir

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False.")

    from huggingface_hub import snapshot_download

    os.makedirs(path, exist_ok=True)
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=path)
    return data_dir


def get_ctspine1k_paths(
    path: Union[os.PathLike, str],
    source: Optional[Literal["colonog", "covid19", "hnscc", "msd_liver"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CTSpine1K data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        source: The choice of source collection. All of them are used if it is not given.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if source is not None and source not in SOURCES:
        raise ValueError(f"'{source}' is not a valid source. Choose from {list(SOURCES.keys())}.")

    data_dir = get_ctspine1k_data(path, download)

    folders = list(SOURCES.values()) if source is None else [SOURCES[source]]
    prepared_dir = os.path.join(path, "prepared")
    raw_paths, label_paths = [], []
    for folder in folders:
        os.makedirs(os.path.join(prepared_dir, folder), exist_ok=True)
        for image_path in natsorted(glob(os.path.join(data_dir, "volumes", folder, "*.nii.gz"))):
            stem = os.path.basename(image_path)[:-len(".nii.gz")]
            label_path = os.path.join(data_dir, "labels", folder, f"{stem}_seg.nii.gz")
            if not os.path.exists(label_path):
                continue
            raw_paths.append(_prepare_name(image_path, prepared_dir, folder, stem))
            label_paths.append(_prepare_name(label_path, prepared_dir, folder, f"{stem}_seg"))

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_ctspine1k_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    source: Optional[Literal["colonog", "covid19", "hnscc", "msd_liver"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CTSpine1K dataset for vertebra segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        source: The choice of source collection. All of them are used if it is not given.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_ctspine1k_paths(path, source, download)

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


def get_ctspine1k_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    source: Optional[Literal["colonog", "covid19", "hnscc", "msd_liver"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CTSpine1K dataloader for vertebra segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        source: The choice of source collection. All of them are used if it is not given.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ctspine1k_dataset(path, patch_shape, source, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
