"""The OASIS dataset contains two set of annotations:
one for 4 tissue segmentation and 35 anatomical segmentation in brain T1 MRI.

The dataset comes from https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md.

This dataset is from the following publications:
- https://doi.org/10.59275/j.melba.2022-74f1
- https://doi.org/10.1162/jocn.2007.19.9.1498

Please cite them if you use this dataset for your research.
"""

import os
from glob import glob
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.v1.0.tar"
CHECKSUM = "86dd117dda17f736ade8a4088d7e98e066e1181950fe8b406f1a35f7fb743e78"


def get_oasis_data(path: Union[os.PathLike, str], download: bool = False):
    """Download the OASIS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.
    """
    data_path = os.path.join(path, "data")
    if os.path.exists(data_path):
        return

    os.makedirs(path, exist_ok=True)
    util.download_source(path=path, url=URL, download=download, checksum=CHECKSUM)
    tar_path = os.path.join(path, "neurite-oasis.v1.0.tar")
    util.unzip_tarfile(tar_path=tar_path, dst=data_path, remove=False)


def get_oasis_paths(
    path: Union[os.PathLike, str],
    source: Literal['orig', 'norm'] = "orig",
    label_annotations: Literal['4', '35'] = "4",
    download: bool = False
) -> Tuple[List[int], List[int]]:
    """Get paths to the OASIS data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        source: The source of inputs. Either 'orig' (original brain scans) or 'norm' (skull stripped).
        label_annotations: The set of annotations. Either '4' (for tissues) or '35' (for anatomy).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_oasis_data(path, download)

    patient_dirs = glob(os.path.join(path, "data", "OASIS_*"))
    raw_paths, label_paths = [], []
    for pdir in patient_dirs:
        raw_paths.append(os.path.join(pdir, f"seg{label_annotations}.nii.gz"))
        label_paths.append(os.path.join(pdir, f"{source}.nii.gz"))

    assert len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def get_oasis_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    source: Literal['orig', 'norm'] = "orig",
    label_annotations: Literal['4', '35'] = "4",
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the OASIS dataset for tissue / anatomical segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        source: The source of inputs. Either 'orig' (original brain scans) or 'norm' (skull stripped).
        label_annotations: The set of annotations. Either '4' (for tissues) or '35' (for anatomy).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_oasis_paths(path, source, label_annotations, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_oasis_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    source: Literal['orig', 'norm'] = "orig",
    label_annotations: Literal['4', '35'] = "4",
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the OASIS dataloader for tissue / anatomical segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        source: The source of inputs. Either 'orig' (original brain scans) or 'norm' (skull stripped).
        label_annotations: The set of annotations. Either '4' (for tissues) or '35' (for anatomy).
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_oasis_dataset(path, patch_shape, source, label_annotations, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
