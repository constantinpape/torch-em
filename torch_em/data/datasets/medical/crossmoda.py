"""The crossMoDA dataset contains annotations for vestibular schwannoma and cochlea segmentation
in contrast-enhanced T1-weighted (ceT1) MRI.

The data comes from the crossMoDA 2022 challenge (https://crossmoda2022.grand-challenge.org/) for unsupervised
cross-modality domain adaptation. The training set contains 210 annotated ceT1 scans (the source domain,
105 from London and 105 from Tilburg) and 210 unpaired, unannotated high-resolution T2 scans (the target domain).
Only the annotated source domain is used for the segmentation dataset and loader, the target domain images
can be accessed via `get_crossmoda_paths` with `domain='target'`.

The label ids are: background: 0, vestibular schwannoma (tumor): 1 and cochlea: 2.

The dataset is located at https://zenodo.org/records/6504722.

This dataset is from the publications https://doi.org/10.7937/TCIA.9YTJ-5Q73 and
https://doi.org/10.1016/j.media.2022.102628.
Please cite them if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/6504722/files/crossmoda2022_training.zip?download=1"
CHECKSUM = "d3db17e04fd7b4c7bfc8cd569f63e01953cc3a218d5bc754f916729483ef0cdb"

# The center is encoded in the filenames: 'ldn' for London and 'etz' for Tilburg.
CENTERS = {"london": "ldn", "tilburg": "etz"}


def get_crossmoda_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the crossMoDA 2022 training data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath to the folder with the downloaded data.
    """
    source_dir = os.path.join(path, "training_source")
    target_dir = os.path.join(path, "training_target")
    if os.path.exists(source_dir) and os.path.exists(target_dir):
        return path

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "crossmoda2022_training.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return path


def get_crossmoda_paths(
    path: Union[os.PathLike, str],
    center: Optional[Literal["london", "tilburg"]] = None,
    domain: Literal["source", "target"] = "source",
    download: bool = False,
) -> Tuple[List[str], Optional[List[str]]]:
    """Get paths to the crossMoDA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        center: The center the scans come from. Either 'london' or 'tilburg'. By default, both centers are used.
        domain: The domain of the scans. Either 'source' (annotated ceT1) or 'target' (unannotated hrT2).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data. None for the 'target' domain, which has no annotations.
    """
    if center is not None and center not in CENTERS:
        raise ValueError(f"'{center}' is not a valid center.")
    if domain not in ("source", "target"):
        raise ValueError(f"'{domain}' is not a valid domain.")

    data_dir = get_crossmoda_data(path, download)

    center_pattern = "*" if center is None else CENTERS[center]
    sequence = "ceT1" if domain == "source" else "hrT2"
    pattern = os.path.join(data_dir, f"training_{domain}", f"crossmoda*_{center_pattern}_*_{sequence}.nii.gz")
    raw_paths = natsorted(glob(pattern))
    assert len(raw_paths) > 0, f"No crossMoDA volumes found at '{pattern}'."

    if domain == "target":
        return raw_paths, None

    label_paths = [p.replace(f"_{sequence}.nii.gz", "_Label.nii.gz") for p in raw_paths]
    assert all(os.path.exists(p) for p in label_paths)

    return raw_paths, label_paths


def get_crossmoda_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    center: Optional[Literal["london", "tilburg"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the crossMoDA dataset for vestibular schwannoma and cochlea segmentation in ceT1 MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        center: The center the scans come from. Either 'london' or 'tilburg'. By default, both centers are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_crossmoda_paths(path, center, "source", download)

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


def get_crossmoda_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    center: Optional[Literal["london", "tilburg"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the crossMoDA dataloader for vestibular schwannoma and cochlea segmentation in ceT1 MRI.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        center: The center the scans come from. Either 'london' or 'tilburg'. By default, both centers are used.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_crossmoda_dataset(path, patch_shape, center, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
