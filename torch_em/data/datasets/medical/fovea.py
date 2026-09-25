"""FOVEA is a dataset for optic disc and retinal vessel segmentation in preoperative and
intraoperative retinal fundus images, comprising 40 patients collected at Moorfields Eye
Hospital (London, UK). For each patient and each domain (preoperative color fundus photography
and intraoperative retinal microscopy), the green channel image used for annotation as well as
binary optic disc and retinal vessel masks from two independent clinical research fellows are
provided.

The dataset is located at https://doi.org/10.6084/m9.figshare.28329338.
This dataset is from the publication https://doi.org/10.1038/s41597-025-04965-2.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://ndownloader.figshare.com/files/52512728"
CHECKSUM = "4f9b88468e79b89d9c561932c28e6bd0974bde1ce2a21b1027b7dca2e09bf24d"

DOMAINS = {"preoperative": "p", "intraoperative": "i"}
ANNOTATIONS = {"optic_disc": "od", "vessels": "ve"}


def get_fovea_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the FOVEA dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    if os.path.exists(path) and glob(os.path.join(path, "FOVEA*_img.png")):
        return path

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "FOVEA.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return path


def get_fovea_paths(
    path: Union[os.PathLike, str],
    domain: Literal["preoperative", "intraoperative", "both"] = "both",
    annotation: Literal["optic_disc", "vessels"] = "vessels",
    annotator: Literal[1, 2, "both"] = 1,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the FOVEA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        domain: The choice of imaging domain. One of 'preoperative', 'intraoperative' or 'both'.
        annotation: The choice of segmentation target. Either 'optic_disc' or 'vessels'.
        annotator: The choice of annotator. Either 1, 2 or 'both' (uses annotations from both annotators).
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_fovea_data(path=path, download=download)

    if domain == "both":
        domain_tags = list(DOMAINS.values())
    elif domain in DOMAINS:
        domain_tags = [DOMAINS[domain]]
    else:
        raise ValueError(f"'{domain}' is not a valid domain. Choose from {list(DOMAINS) + ['both']}.")

    if annotation not in ANNOTATIONS:
        raise ValueError(f"'{annotation}' is not a valid annotation. Choose from {list(ANNOTATIONS)}.")
    annotation_tag = ANNOTATIONS[annotation]

    if annotator == "both":
        annotators = [1, 2]
    elif annotator in (1, 2):
        annotators = [annotator]
    else:
        raise ValueError(f"'{annotator}' is not a valid annotator. Choose from 1, 2 or 'both'.")

    image_paths, label_paths = [], []
    for domain_tag in domain_tags:
        for ann in annotators:
            label_glob = natsorted(glob(os.path.join(data_dir, f"FOVEA*_{domain_tag}_{annotation_tag}_{ann}.png")))
            for label_path in label_glob:
                image_path = label_path.replace(f"_{annotation_tag}_{ann}.png", "_img.png")
                assert os.path.exists(image_path), f"The image at '{image_path}' does not exist."
                image_paths.append(image_path)
                label_paths.append(label_path)

    assert len(image_paths) == len(label_paths) and len(image_paths) > 0

    return image_paths, label_paths


def get_fovea_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    domain: Literal["preoperative", "intraoperative", "both"] = "both",
    annotation: Literal["optic_disc", "vessels"] = "vessels",
    annotator: Literal[1, 2, "both"] = 1,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the FOVEA dataset for optic disc and retinal vessel segmentation in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        domain: The choice of imaging domain.
        annotation: The choice of segmentation target.
        annotator: The choice of annotator.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, label_paths = get_fovea_paths(path, domain, annotation, annotator, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_fovea_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    domain: Literal["preoperative", "intraoperative", "both"] = "both",
    annotation: Literal["optic_disc", "vessels"] = "vessels",
    annotator: Literal[1, 2, "both"] = 1,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the FOVEA dataloader for optic disc and retinal vessel segmentation in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        domain: The choice of imaging domain.
        annotation: The choice of segmentation target.
        annotator: The choice of annotator.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_fovea_dataset(path, patch_shape, domain, annotation, annotator, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
