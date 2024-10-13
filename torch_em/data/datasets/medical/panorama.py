"""The PANORAMA dataset contains annotation for PDAC lesion, veins, arteries, pancreas parenchyma,
pancreatic duct and common bile duct segmentation in CT scans.

The dataset is from the PANORAMA challenge: https://panorama.grand-challenge.org/.

NOTE: The latest information for the label legends are located at:
https://github.com/DIAGNijmegen/panorama_labels#label-legend.
The label legends are described as follows:
- background: 0
- PDAC lesion: 1
- veins: 2
- arteries: 3
- pancreas parenchyma: 4
- pancreatic duct: 5
- common bile duct: 6

This dataset is from the article: https://doi.org/10.5281/zenodo.10599559
Please cite it if you use this dataset in your research.
"""

import os
import shutil
import subprocess
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Optional, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "batch_1": "https://zenodo.org/records/13715870/files/batch_1.zip",
    "batch_2": "https://zenodo.org/records/13742336/files/batch_2.zip",
    "batch_3": "https://zenodo.org/records/11034011/files/batch_3.zip",
    "batch_4": "https://zenodo.org/records/10999754/files/batch_4.zip",
}

CHECKSUMS = {
    "batch_1": "aff39b6347650d6c7457adf7a04bfb0a651ab6ecd33676ff109bdab17bc41cff",
    "batch_2": "db6353a2c1c565c8bf084bd4fe1512fd6020b7675a1c9ab61b9a13d72a9fe76c",
    "batch_3": "c1d71b40948edc36f795a7801cc79000082df8d365c48574af50b36516d64cee",
    "batch_4": "3b5341af79c2cc8b8a9fa3ab7a6cfa8fedf694538a3d6be97c18e5c82be4d9d8",
}


def get_panorama_data(path: Union[os.PathLike, str], download: bool = False):
    """Download the PANORAMA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.
    """
    os.makedirs(path, exist_ok=True)

    data_path = os.path.join(path, "volumes")
    label_path = os.path.join(path, "labels")
    if os.path.exists(data_path) and os.path.exists(label_path):
        return

    print("PANORAMA is a large dataset. I might take a while to download the volumes and respective labels.")

    # Download the label volumes.
    subprocess.call(
        ["git", "clone", "--quiet", "https://github.com/DIAGNijmegen/panorama_labels", label_path]
    )

    def _move_batch_data_to_root(batch):
        if batch in ["batch_3", "batch_4"]:
            batch_dir = os.path.join(data_path, batch)

            for fpath in glob(os.path.join(batch_dir, "*.nii.gz")):
                shutil.move(src=fpath, dst=data_path)

            if os.path.exists(batch_dir):
                shutil.rmtree(batch_dir)

    # Download the input volumes.
    for batch in URLS.keys():
        zip_path = os.path.join(path, f"{batch}.zip")
        util.download_source(path=zip_path, url=URLS[batch], download=download, checksum=CHECKSUMS[batch])
        util.unzip(zip_path=zip_path, dst=data_path)
        _move_batch_data_to_root(batch)


def get_panorama_paths(
    path: Union[os.PathLike, str],
    annotation_choice: Optional[Literal["manual", "automatic"]] = None,
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the PANORAMA data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        annotation_choice: The source of annotation.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_panorama_data(path, download)

    if annotation_choice is None:
        annotation_choice = "*"
    label_paths = natsorted(glob(os.path.join(path, "labels", f"{annotation_choice}_labels", "*.nii.gz")))
    raw_dir = os.path.join(path, "volumes")
    raw_paths = [
        os.path.join(raw_dir, os.path.basename(fpath).replace(".nii.gz", "_0000.nii.gz")) for fpath in label_paths
    ]

    # NOTE: the label "100051_00001.nii.gz" returns the error: 'nibabel.filebasedimages.ImageFileError: Empty file'
    # We simply do not consider the sample (and correspondign labels) for the dataset.
    for rpath, lpath in zip(raw_paths, label_paths):
        if rpath.find("100051_00001") != -1:
            raw_paths.remove(rpath)

        if lpath.find("100051_00001") != -1:
            label_paths.remove(lpath)

    assert len(raw_paths) == len(label_paths)

    return raw_paths, label_paths


def get_panorama_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    annotation_choice: Optional[Literal["manual", "automatic"]] = None,
    download: bool = False, **kwargs
) -> Dataset:
    """Get the PANORAMA dataset for pancreatic lesion (and other structures) segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        annotation_choice: The source of annotation.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_panorama_paths(path, annotation_choice, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        is_seg_dataset=True,
        patch_shape=patch_shape,
        **kwargs
    )


def get_panorama_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    annotation_choice: Optional[Literal["manual", "automatic"]] = None,
    download: bool = False, **kwargs
) -> DataLoader:
    """Get the PANORAMA dataloader for pancreatic lesion (and other structures) segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        annotation_choice: The source of annotation.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_panorama_dataset(path, patch_shape, annotation_choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
