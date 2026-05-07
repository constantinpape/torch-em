"""The CVZ-Fluo dataset contains annotations for cell and nuclei segmentation in
fluorescence microscopy images.

The dataset is from the publication https://doi.org/10.1038/s41597-023-02108-z.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Literal, Tuple, Optional, List

import imageio.v3 as imageio
from skimage.measure import label as connected_components

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util
from .neurips_cell_seg import to_rgb


URL = "https://www.synapse.org/Synapse:syn27624812/"


def get_cvz_fluo_data(path: Union[os.PathLike, str], download: bool = False):
    """Download the CVZ-Fluo dataset.

    Args:
        path: Filepath to a folder where the downloaded data is saved.
        download: Whether to download the data if it is not present.
    """
    data_dir = os.path.join(path, r"Annotation Panel Table.xlsx")
    if not os.path.exists(data_dir):
        os.makedirs(path, exist_ok=True)
        # Download the dataset from 'synapse'.
        util.download_source_synapse(path=path, entity="syn27624812", download=download)

    return


def _preprocess_labels(label_paths):
    neu_label_paths, to_process = [], []

    # First, make simple checks to avoid redundant progress bar runs.
    for lpath in label_paths:
        neu_lpath = lpath.replace(".png", ".tif")
        neu_label_paths.append(neu_lpath)

        if not os.path.exists(neu_lpath):
            to_process.append((lpath, neu_lpath))

    if to_process:  # Next, process valid inputs.
        for lpath, neu_lpath in tqdm(to_process, desc="Preprocessing labels"):
            if not os.path.exists(lpath):  # HACK: Some paths have weird spacing nomenclature.
                lpath = Path(lpath).parent / rf" {os.path.basename(lpath)}"

            label = imageio.imread(lpath)
            imageio.imwrite(neu_lpath, connected_components(label).astype(label.dtype), compression="zlib")

    return neu_label_paths


def get_cvz_fluo_paths(
    path: Union[os.PathLike, str],
    stain_choice: Literal["cell", "dapi"],
    data_choice: Optional[Literal["CODEX", "Vectra", "Zeiss"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the CVZ-Fluo data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_cvz_fluo_data(path, download)

    if data_choice is None:
        data_choice = "**"
    else:
        if data_choice == "Zeiss" and stain_choice == "dapi":
            raise ValueError("'Zeiss' data does not have DAPI stained images.")

        data_choice = f"{data_choice}/**"

    if stain_choice not in ["cell", "dapi"]:
        raise ValueError(f"'{stain_choice}' is not a valid stain choice.")

    raw_paths = natsorted(
        glob(os.path.join(path, data_choice, f"*-Crop_{stain_choice.title()}_Png.png"), recursive=True)
    )
    label_paths = [p.replace("_Png.png", "_Mask_Png.png") for p in raw_paths]
    label_paths = _preprocess_labels(label_paths)

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    return raw_paths, label_paths


def get_cvz_fluo_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    stain_choice: Literal["cell", "dapi"],
    data_choice: Optional[Literal["CODEX", "Vectra", "Zeiss"]] = None,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the CVZ-Fluo dataset for cell and nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        stain_choice: Decides for annotations based on staining. Either "cell" (for cells) or "dapi" (for nuclei).
        data_choice: The choice of dataset.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_cvz_fluo_paths(path, stain_choice, data_choice, download)

    if "raw_transform" not in kwargs:
        kwargs["raw_transform"] = torch_em.transform.get_raw_transform(augmentation2=to_rgb)

    if "transform" not in kwargs:
        kwargs["transform"] = torch_em.transform.get_augmentations(ndim=2)

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_cvz_fluo_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    stain_choice: Literal["cell", "dapi"],
    data_choice: Optional[Literal["CODEX", "Vectra", "Zeiss"]] = None,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the CVZ-Fluo dataloader for cell and nucleus segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training
        patch_shape: The patch shape to use for training.
        stain_choice: Decides for annotations based on staining. Either "cell" (for cells) or "dapi" (for nuclei).
        data_choice: The choice of dataset.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cvz_fluo_dataset(path, patch_shape, stain_choice, data_choice, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
