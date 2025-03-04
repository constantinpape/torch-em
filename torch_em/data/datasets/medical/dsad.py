"""The DSAD dataset contains annotations for abdominal organs in laparoscopy images.

This dataset is located at https://springernature.figshare.com/articles/dataset/The_Dresden_Surgical_Anatomy_Dataset_for_abdominal_organ_segmentation_in_surgical_data_science/21702600  # noqa
The dataset is from the publication https://doi.org/10.1038/s41597-022-01719-2.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://springernature.figshare.com/ndownloader/files/38494425"
CHECKSUM = "b8a8ade37d106fc1641a901d1c843806f2d27f9f8e18f4614b043e7e2ca2e40f"
ORGANS = [
    "abdominal_wall", "inferior_mesenteric_artery", "liver", "pancreas", "spleen", "ureter",
    "colon", "intestinal_veins", "multilabel", "small_intestine", "stomach", "vesicular_glands"
]


def get_dsad_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the DSAD dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "data.zip")
    print("Downloading the DSAD data. Might take several minutes depending on your internet connection.")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=os.path.join(path, "data"), remove=False)
    print("The download has finished and the data has been unzipped to a target folder.")

    return data_dir


def get_dsad_paths(
    path: Union[os.PathLike, str], organ: Optional[str] = None, download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the DSAD data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        organ: The choice of organ annotations.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_dsad_data(path, download)

    if organ is None:
        organ = "*"
    else:
        assert organ in ORGANS, f"'{organ}' is not a valid organ choice."
        assert isinstance(organ, str), "We currently support choosing one organ at a time."

    image_paths = natsorted(glob(os.path.join(data_dir, organ, "*", "image*.png")))
    # Remove multi-label inputs.
    image_paths = [p for p in image_paths if "multilabel" not in p]

    # Get label paths.
    mask_paths = [p.replace("image", "mask") for p in image_paths]
    assert all([os.path.exists(p) for p in mask_paths])

    assert image_paths and len(image_paths) == len(mask_paths)

    return image_paths, mask_paths


def get_dsad_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    organ: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the DSAD dataset for organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        organ: The choice of organ annotations.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, mask_paths = get_dsad_paths(path, organ, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=mask_paths,
        label_key=None,
        patch_shape=patch_shape,
        with_channels=True,
        is_seg_dataset=False,
        **kwargs
    )


def get_dsad_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    organ: Optional[str] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the DSAD dataloader for organ segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        organ: The choice of organ annotations.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The Dataloader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_dsad_dataset(path, patch_shape, organ, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
