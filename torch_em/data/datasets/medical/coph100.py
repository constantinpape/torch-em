"""COph100 is a dataset for fundus image registration from infants, with automatic
vessel segmentation masks provided for each retinal fundus image.

The raw fundus images are a subset of the "Retinal Image Dataset of Infants and
Retinopathy of Prematurity" (RIDIRP), published in Timkovic et al. -
https://doi.org/10.1038/s41597-024-03409-7. COph100 adds manually labeled corresponding
point pairs for registration and automatic vessel segmentation masks on top of this subset.

This dataset is from the publication https://doi.org/10.1038/s41597-025-04426-w.
Please cite it (and the original RIDIRP publication above) if you use this dataset for your research.

NOTE: The COph100 masks are licensed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/), and the underlying RIDIRP fundus
images are licensed under CC0 (https://creativecommons.org/publicdomain/zero/1.0/).
"""

import os
import re
import shutil
from glob import glob
from typing import Union, Tuple, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


COPH100_URL = "https://ndownloader.figshare.com/files/51235925"
COPH100_CHECKSUM = "407ca917280e4f5395b236ffd57096b7ddb2ff3ec362fe150c8ff47c6468e639"

RIDIRP_URL = "https://ndownloader.figshare.com/files/43152595"
RIDIRP_CHECKSUM = "c07f82340210e8a99bffca99a270a5e6d2c00bb5f1b75e3bc7dac98ad4696163"


def _copy_raw_images_from_ridirp(coph100_dir, ridirp_dir):
    # The COph100 archive only ships point annotations and vessel masks. The raw fundus
    # images have to be copied over from the original RIDIRP images, matched by patient
    # id (first three characters of the filename) and examination stage (parsed from the
    # "S<stage>" token in the filename), following the logic of the official
    # "Copy_COph100_from_ROP.py" script shipped alongside the COph100 archive.
    mask_paths = sorted(glob(os.path.join(coph100_dir, "*", "*_mask.png")))
    for mask_path in mask_paths:
        fname = os.path.basename(mask_path)[:-len("_mask.png")]
        patient_id = fname[:3]
        stage = int(re.search(r"S(\d+)", fname).group(1))

        source_path = os.path.join(ridirp_dir, "images", patient_id, f"{stage:02d}", f"{fname}.jpg")
        target_path = os.path.join(os.path.dirname(mask_path), f"{fname}.jpg")

        if os.path.exists(target_path):
            continue

        if not os.path.exists(source_path):
            raise RuntimeError(f"Could not find the expected raw image at '{source_path}'.")

        shutil.copy2(source_path, target_path)


def get_coph100_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the COph100 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    data_dir = os.path.join(path, "COph100")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    coph100_zip_path = os.path.join(path, "COph100.zip")
    util.download_source(path=coph100_zip_path, url=COPH100_URL, download=download, checksum=COPH100_CHECKSUM)
    util.unzip(zip_path=coph100_zip_path, dst=data_dir)

    ridirp_dir = os.path.join(path, "RIDIRP")
    ridirp_zip_path = os.path.join(path, "RIDIRP.zip")
    util.download_source(path=ridirp_zip_path, url=RIDIRP_URL, download=download, checksum=RIDIRP_CHECKSUM)
    util.unzip(zip_path=ridirp_zip_path, dst=ridirp_dir)

    _copy_raw_images_from_ridirp(coph100_dir=data_dir, ridirp_dir=ridirp_dir)

    return data_dir


def get_coph100_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the COph100 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_coph100_data(path=path, download=download)

    gt_paths = sorted(glob(os.path.join(data_dir, "*", "*_mask.png")))
    image_paths = [p[:-len("_mask.png")] + ".jpg" for p in gt_paths]

    return image_paths, gt_paths


def get_coph100_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the COph100 dataset for retinal vessel segmentation in infant fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_coph100_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_coph100_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the COph100 dataloader for retinal vessel segmentation in infant fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_coph100_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
