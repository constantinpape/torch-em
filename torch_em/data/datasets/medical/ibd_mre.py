"""The IBD-MRE dataset contains annotations for bowel segment segmentation in
magnetic resonance enterography (MRE) scans of patients with inflammatory bowel disease (IBD).

The dataset contains coronal HASTE (half-Fourier acquisition single-shot turbo spin-echo) MRE
sequences from 114 IBD patients, with fine pixel-level annotations for ten bowel segments,
labeled by experienced radiologists. The semantic label ids are:
1: stomach, 2: duodenum, 3: small intestine, 4: appendix, 5: cecum, 6: ascending colon,
7: transverse colon, 8: descending colon, 9: sigmoid colon, 10: rectum.

The dataset is located at https://doi.org/10.5281/zenodo.13839321.
This dataset is from the publication https://doi.org/10.1038/s41597-025-04760-z.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/13839321/files/A%20comprehensive%20dataset.zip"
CHECKSUM = "d88541e64f33629b8390addd286f7dea88e665d29c2cec21d7594b7c64c97393"


def get_ibd_mre_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the IBD-MRE dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "A comprehensive dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def get_ibd_mre_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the IBD-MRE data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_ibd_mre_data(path=path, download=download)

    image_paths = sorted(
        glob(os.path.join(data_dir, "*_data.nii.gz")), key=lambda p: int(os.path.basename(p).split("_")[0])
    )
    gt_paths = sorted(
        glob(os.path.join(data_dir, "*_label.nii.gz")), key=lambda p: int(os.path.basename(p).split("_")[0])
    )

    return image_paths, gt_paths


def get_ibd_mre_dataset(
    path: Union[os.PathLike, str], patch_shape: Tuple[int, ...], download: bool = False, **kwargs
) -> Dataset:
    """Get the IBD-MRE dataset for segmentation of bowel segments in MRE scans.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_ibd_mre_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_ibd_mre_loader(
    path: Union[os.PathLike, str], batch_size: int, patch_shape: Tuple[int, ...], download: bool = False, **kwargs
) -> DataLoader:
    """Get the IBD-MRE dataloader for segmentation of bowel segments in MRE scans.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ibd_mre_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
