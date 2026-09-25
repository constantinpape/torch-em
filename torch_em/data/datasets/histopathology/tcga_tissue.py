"""The TCGA Tissue Segmentation dataset contains annotations for tissue vs. background
segmentation in histopathology slide images drawn from The Cancer Genome Atlas (TCGA).

Each image is a full TCGA slide (predominantly H&E stained, some FFPE and frozen sections)
downsampled to 10 micrometer per pixel. The masks are binary, with tissue regions labeled as
foreground, including artifact-affected regions such as pen markings, ink, air bubbles and cracks.

The dataset is located at https://huggingface.co/datasets/conflux-xyz/tcga-tissue-segmentation
and is licensed under CC0-1.0. It is not associated with a peer-reviewed publication. Users are
asked to cite the TCGA Research Network: https://www.cancer.gov/tcga.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


HF_REPO = "conflux-xyz/tcga-tissue-segmentation"


def get_tcga_tissue_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the TCGA Tissue Segmentation dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the folder where the data is stored.
    """
    if os.path.exists(os.path.join(path, "images")) and os.path.exists(os.path.join(path, "masks")):
        return path

    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but 'download' is set to False.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

    os.makedirs(path, exist_ok=True)
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=path)

    return path


def get_tcga_tissue_paths(
    path: Union[os.PathLike, str], split: Literal["train", "test"], download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the TCGA Tissue Segmentation data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data split, either 'train' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_tcga_tissue_data(path, download)

    split_path = os.path.join(data_dir, f"{split}-slides.txt")
    with open(split_path) as f:
        slide_ids = {line.strip() for line in f if line.strip()}

    raw_paths = natsorted(
        p for p in glob(os.path.join(data_dir, "images", "*.png"))
        if os.path.splitext(os.path.basename(p))[0] in slide_ids
    )
    label_paths = natsorted(
        p for p in glob(os.path.join(data_dir, "masks", "*.png"))
        if os.path.splitext(os.path.basename(p))[0] in slide_ids
    )

    assert len(raw_paths) == len(label_paths) == len(slide_ids)
    assert all(
        os.path.basename(raw_path) == os.path.basename(label_path)
        for raw_path, label_path in zip(raw_paths, label_paths)
    )

    return raw_paths, label_paths


def get_tcga_tissue_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the TCGA Tissue Segmentation dataset for tissue vs. background segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split, either 'train' or 'test'.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_tcga_tissue_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        ndim=2,
        with_channels=True,
        **kwargs,
    )


def get_tcga_tissue_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"] = "train",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the TCGA Tissue Segmentation dataloader for tissue vs. background segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split, either 'train' or 'test'.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_tcga_tissue_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
