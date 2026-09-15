"""The BBBC024 dataset contains synthetic 3D fluorescence microscopy images of HL60 cell nuclei
with instance segmentation ground truth.

The images were generated with a virtual microscope (CytoPacq) imitating a Zeiss S100 confocal
microscope. Each image contains 20 nuclei and has a shape of (129, 565, 807) voxels (ZYX).
The dataset is organized in 8 subsets of 30 images each: the nuclei cluster with a probability
of 0%, 25%, 50% or 75%, and every clustering level is provided in a low SNR and a high SNR variant.
The ground truth is a 16-bit labeled volume with the ids 1 to 20 for the individual nuclei and 0 for background.

The dataset is located at https://bbbc.broadinstitute.org/BBBC024.
This dataset is from the publication https://doi.org/10.1002/cyto.a.20714.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Literal, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://data.broadinstitute.org/bbbc/BBBC024/BBBC024_v1_{subset}_images_TIFF.zip"
CHECKSUMS = {
    "c00_lowSNR": "c725828c267c347245e94f2dd4bfd65222fbbb526eb020fd5143cf7c7400e495",
    "c00_highSNR": "2a18179503dbd00172c6ea62a7e55752a5e86816b44faf8d1abc8f4cc9346c7b",
    "c25_lowSNR": "7a6a302ab8335657dfb98d8d129691204f333b87f00abdedc5737cc27c80101a",
    "c25_highSNR": "83662cad96f01d6956eba40a3b4b1d1e05f90400ff4fb63406d6bcebbac63a6b",
    "c50_lowSNR": "80f1b11f985fae6052619083c2a8bff0dc6a4a75c1beb2d7b2d90257f59c69f7",
    "c50_highSNR": "ba8fe5d26adec8d6e7ab224b06c1261218825ab1b3bf260a30515d6ebf77fa21",
    "c75_lowSNR": "b0e2f19d70012e2be4ccf7a7f1d0502858a45d60d1af01d2e32fad7c86263af3",
    "c75_highSNR": "2d56055a09d7dd22911f593dcfdef93be074ae27d58d7f7d32d089a25217fc09",
}


def _get_subset_name(clustering, snr):
    if clustering not in (0, 25, 50, 75):
        raise ValueError(f"'{clustering}' is not a valid clustering probability. Choose from 0, 25, 50 or 75.")
    if snr not in ("low", "high"):
        raise ValueError(f"'{snr}' is not a valid SNR level. Choose from 'low' or 'high'.")
    return f"c{clustering:02d}_{snr}SNR"


def get_bbbc024_data(
    path: Union[os.PathLike, str],
    clustering: Literal[0, 25, 50, 75] = 0,
    snr: Literal["low", "high"] = "high",
    download: bool = False,
) -> str:
    """Download the BBBC024 dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        clustering: The clustering probability of the nuclei in percent. One of 0, 25, 50 or 75.
        snr: The signal-to-noise ratio of the images. Either 'low' or 'high'.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data for the chosen subset is stored.
    """
    subset = _get_subset_name(clustering, snr)
    data_dir = os.path.join(path, "BBBC024", subset)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, f"BBBC024_v1_{subset}_images_TIFF.zip")
    util.download_source(zip_path, URL.format(subset=subset), download, CHECKSUMS[subset])
    util.unzip(zip_path, data_dir)

    return data_dir


def get_bbbc024_paths(
    path: Union[os.PathLike, str],
    clustering: Literal[0, 25, 50, 75] = 0,
    snr: Literal["low", "high"] = "high",
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the BBBC024 data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        clustering: The clustering probability of the nuclei in percent. One of 0, 25, 50 or 75.
        snr: The signal-to-noise ratio of the images. Either 'low' or 'high'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_bbbc024_data(path, clustering, snr, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "image-final_*.tif")))
    label_paths = natsorted(glob(os.path.join(data_dir, "image-labels_*.tif")))
    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_bbbc024_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    clustering: Literal[0, 25, 50, 75] = 0,
    snr: Literal["low", "high"] = "high",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the BBBC024 dataset for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        clustering: The clustering probability of the nuclei in percent. One of 0, 25, 50 or 75.
        snr: The signal-to-noise ratio of the images. Either 'low' or 'high'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_bbbc024_paths(path, clustering, snr, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_bbbc024_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    clustering: Literal[0, 25, 50, 75] = 0,
    snr: Literal["low", "high"] = "high",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the BBBC024 dataloader for nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        clustering: The clustering probability of the nuclei in percent. One of 0, 25, 50 or 75.
        snr: The signal-to-noise ratio of the images. Either 'low' or 'high'.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_bbbc024_dataset(path, patch_shape, clustering, snr, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
