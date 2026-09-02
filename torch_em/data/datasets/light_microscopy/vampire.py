"""The VAMPIRE dataset contains annotations for cell segmentation in
fluorescence microscopy images of mouse embryonic fibroblasts.

The cells are stained with phalloidin and imaged in a single channel at 1024 x 1280 pixels.
They come in two conditions, wildtype cells and cells with a Lamin A/C knockout, with 30 images
each. The masks were produced with the CellProfiler pipeline that the authors ship next to the
images, and they label whole cells.

NOTE: The annotations are sparse. The CellProfiler pipeline drops objects that touch the image
border and objects outside its size filters, so roughly a third of the cells in the wildtype images
and roughly half of the cells in the knockout images carry no mask. Unlabeled cells are part of the
background, so a sampler or a loss that ignores the background is advisable.

NOTE: The study deposits two further image sets that carry no masks and are therefore not part of
this loader. Mouse embryonic fibroblasts on micropatterns are at
https://github.com/kukionfr/Micropattern_MEF_LMNA_Image and human dermal fibroblast nuclei of seven
donor ages are at https://github.com/kukionfr/Aging_human_dermal_fibroblast_nucleus . Both hold
morphometric tables of the cells rather than segmentations.

The data is published as the supplementary data of the VAMPIRE software at
https://github.com/kukionfr/VAMPIRE_open . This dataset is from the publication
https://doi.org/10.1038/s41596-020-00432-x .
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import List, Tuple, Union, Literal, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://github.com/kukionfr/VAMPIRE_open/releases/download/v1.0/Supplementary.Data.zip"
CHECKSUM = "a5e9b70537d5add8b860fc8ac4b40c7f7e260dfd21437926562c41a6d8f95da7"

# The folder names of the two conditions in the archive.
SAMPLE_TYPES = {"wildtype": "MEF_wildtype", "lmna_knockout": "MEF_LMNA--"}


def _preprocess_labels(data_dir: str) -> str:
    """Map the masks to consecutive instance ids.

    The archive stores the instance ids spread over the full uint16 range, so that the masks display
    well in an image viewer. This restores the ids that the loader expects.
    """
    import numpy as np
    import imageio.v3 as imageio
    from tqdm import tqdm

    output_dir = os.path.join(data_dir, "preprocessed")

    for sample_type, folder in SAMPLE_TYPES.items():
        label_dir = os.path.join(output_dir, sample_type)
        os.makedirs(label_dir, exist_ok=True)

        label_paths = natsorted(glob(os.path.join(data_dir, "Example segmented images", folder, "*.tiff")))
        if not label_paths:
            raise RuntimeError(f"Could not find any masks for '{sample_type}' in {data_dir}.")

        for label_path in tqdm(label_paths, desc=f"Preprocess '{sample_type}'"):
            output_path = os.path.join(label_dir, f"{os.path.splitext(os.path.basename(label_path))[0]}.tif")
            if os.path.exists(output_path):
                continue

            labels = imageio.imread(label_path)
            ids = np.unique(labels)
            relabeled = np.searchsorted(ids, labels).astype("uint16")
            if ids[0] != 0:  # Guard against a mask without background.
                relabeled += 1

            temporary_path = f"{output_path}.tmp.tif"
            imageio.imwrite(temporary_path, relabeled, compression="zlib")
            os.replace(temporary_path, output_path)

    return output_dir


def get_vampire_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the VAMPIRE dataset.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the extracted data.
    """
    data_dir = os.path.join(path, "Supplementary Data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "Supplementary.Data.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def get_vampire_paths(
    path: Union[os.PathLike, str],
    sample_type: Optional[Literal["wildtype", "lmna_knockout"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the VAMPIRE data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        sample_type: The condition of the cells. Either 'wildtype' or 'lmna_knockout'. By default, both are used.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the image data.
        List of filepaths to the label data.
    """
    if sample_type is None:
        sample_types = list(SAMPLE_TYPES)
    elif sample_type in SAMPLE_TYPES:
        sample_types = [sample_type]
    else:
        raise ValueError(f"'{sample_type}' is not a valid sample type. Choose from {list(SAMPLE_TYPES)}.")

    data_dir = get_vampire_data(path, download)
    output_dir = _preprocess_labels(data_dir)

    image_paths, label_paths = [], []
    for name in sample_types:
        for image_path in natsorted(glob(os.path.join(data_dir, "Example images", SAMPLE_TYPES[name], "*.tif"))):
            # The archive gives an image and its mask the same file name.
            label_path = os.path.join(output_dir, name, os.path.basename(image_path))
            if not os.path.exists(label_path):
                raise RuntimeError(f"Could not find the mask for the image '{image_path}' at {label_path}.")
            image_paths.append(image_path)
            label_paths.append(label_path)

    if not image_paths:
        raise RuntimeError(f"Could not find any VAMPIRE data in {data_dir}.")

    return image_paths, label_paths


def get_vampire_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    sample_type: Optional[Literal["wildtype", "lmna_knockout"]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the VAMPIRE dataset for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        sample_type: The condition of the cells. Either 'wildtype' or 'lmna_knockout'. By default, both are used.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_vampire_paths(path, sample_type, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, offsets=offsets, boundaries=boundaries, binary=binary,
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs,
    )


def get_vampire_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    sample_type: Optional[Literal["wildtype", "lmna_knockout"]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the VAMPIRE dataloader for cell segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        sample_type: The condition of the cells. Either 'wildtype' or 'lmna_knockout'. By default, both are used.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_vampire_dataset(
        path=path,
        patch_shape=patch_shape,
        sample_type=sample_type,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
