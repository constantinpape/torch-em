"""The PolypGen dataset contains annotations for polyp detection and segmentation in
colonoscopy images and video sequence frames, collected from six different medical
centres across Europe and Africa.

NOTE: The dataset is hosted on Synapse. Downloading it requires the 'synapseclient'
python library and a Synapse account with an authentication token stored in the
'~/.synapseConfig' file. See 'get_polypgen_data' for details.

The Synapse project 'syn26376615' has two versions of the multi-centre archive
('PolypGen2021_MultiCenterData_v2.zip' and '..._v3.zip'); this module downloads both
(via 'download_source_synapse') but only extracts and uses the newer v3 archive, which
unpacks into a 'data_C<i>/images_C<i>' and 'data_C<i>/masks_C<i>' folder per centre
(confirmed by inspecting the real archive contents, not assumed).

The dataset is from the publication https://doi.org/10.1038/s41597-023-01981-y.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, List, Optional, Literal

import numpy as np
import imageio.v3 as imageio

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


ENTITY = "syn26376615"

CENTERS = ["C1", "C2", "C3", "C4", "C5", "C6"]


def get_polypgen_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the PolypGen dataset.

    The dataset is located at https://www.synapse.org/Synapse:syn26376615.

    Follow the instructions below to get access to the dataset.
    - Create a free account at https://www.synapse.org.
    - Generate a personal access token and store it in a '~/.synapseConfig' file, see
      https://python-docs.synapse.org/tutorials/authentication/ for details.
    - Install the 'synapseclient' python library.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    extracted_dir = os.path.join(path, "PolypGen2021_MultiCenterData_v3")
    if os.path.exists(extracted_dir):
        return path

    os.makedirs(path, exist_ok=True)
    util.download_source_synapse(path=path, entity=ENTITY, download=download)

    zip_path = os.path.join(path, "PolypGen2021_MultiCenterData_v3.zip")
    util.unzip(zip_path=zip_path, dst=path, remove=False)

    return path


def get_polypgen_paths(
    path: Union[os.PathLike, str],
    center: Optional[Literal["C1", "C2", "C3", "C4", "C5", "C6"]] = None,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the PolypGen data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        center: The choice of a specific centre's data. By default, loads data from all centres.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_polypgen_data(path, download)

    centers = CENTERS if center is None else [center]

    image_paths, gt_paths = [], []
    for this_center in centers:
        this_image_paths = natsorted(
            glob(os.path.join(data_dir, "**", f"images_{this_center}", "*.jpg"), recursive=True)
        )
        for image_path in this_image_paths:
            # Masks live in a sibling 'masks_C<i>' folder and their filename has an extra '_mask' suffix,
            # e.g. 'images_C1/100H0050.jpg' pairs with 'masks_C1/100H0050_mask.jpg'.
            mask_dir = os.path.dirname(image_path).replace(f"images_{this_center}", f"masks_{this_center}")
            gt_path = os.path.join(mask_dir, f"{Path(image_path).stem}_mask.jpg")
            if not os.path.exists(gt_path):
                continue

            image_paths.append(image_path)
            gt_paths.append(gt_path)

    # The masks are lossily JPEG-compressed grayscale images replicated across 3 channels (background
    # near 0, foreground near 255), not the single-channel binary masks 'ImageCollectionDataset' expects.
    # They are binarized once here and cached as '.tif' files next to the original masks.
    neu_gt_paths = []
    for gt_path in tqdm(gt_paths, desc="Preprocessing PolypGen masks"):
        neu_gt_path = os.path.join(os.path.dirname(gt_path), f"{Path(gt_path).stem}.tif")
        neu_gt_paths.append(neu_gt_path)
        if os.path.exists(neu_gt_path):
            continue

        gt = imageio.imread(gt_path)
        if gt.ndim == 3:  # Some centres' masks are single-channel grayscale rather than RGB.
            gt = np.mean(gt, axis=-1)
        gt = (gt > 128).astype("uint8")
        imageio.imwrite(neu_gt_path, gt, compression="zlib")
    gt_paths = neu_gt_paths

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0, (
        "No image-mask pairs were found. The expected per-centre 'images_C<i>' / 'masks_C<i>' folder layout "
        "may not match the actual structure of the downloaded PolypGen data. Please inspect the data at "
        f"'{data_dir}' and update the search pattern in 'get_polypgen_paths' accordingly."
    )

    return image_paths, gt_paths


def get_polypgen_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    center: Optional[Literal["C1", "C2", "C3", "C4", "C5", "C6"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the PolypGen dataset for polyp segmentation in colonoscopy images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        center: The choice of a specific centre's data. By default, loads data from all centres.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_polypgen_paths(path, center, download)

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


def get_polypgen_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    center: Optional[Literal["C1", "C2", "C3", "C4", "C5", "C6"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the PolypGen dataloader for polyp segmentation in colonoscopy images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        center: The choice of a specific centre's data. By default, loads data from all centres.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_polypgen_dataset(path, patch_shape, center, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
