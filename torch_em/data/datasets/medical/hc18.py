"""HC18 is a dataset for segmentation of the fetal head in 2d ultrasound images, together with
annotations for measuring the head circumference (HC).

The dataset is located at https://doi.org/10.5281/zenodo.1327317, released under a CC BY 4.0 license.
This dataset is from the publication https://doi.org/10.1371/journal.pone.0200412. Please cite it if
you use this dataset in your research.

NOTE: The dataset ships pre-rendered annotations as the ellipse fit of the head circumference drawn as
a thin closed contour (not a filled mask). We rasterize a filled binary mask from this contour by
filling the enclosed region (analogous in spirit to polygon rasterization, see `rasterize_rtstruct`
in `torch_em/data/datasets/util.py`, although here the shape is defined by a closed contour rather
than by polygon vertices).

NOTE: Only the training split ships with ground truth annotations. The test split (available at
the same Zenodo record) only provides the raw images and pixel size metadata, without annotations,
and is therefore not supported by this loader.
"""

import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List

import imageio.v3 as imageio
from scipy.ndimage import binary_fill_holes

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/1327317/files/training_set.zip"
CHECKSUM = "fd20d7909df892cfbdc0850de18072dbdad4dc3bc0a57202d4cc818d4715de36"


def get_hc18_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the HC18 dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "training_set")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "training_set.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def _create_mask(annotation_path, gt_path):
    contour = imageio.imread(annotation_path)
    mask = binary_fill_holes(contour > 0).astype("uint8")
    imageio.imwrite(gt_path, mask)


def get_hc18_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the HC18 data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_hc18_data(path=path, download=download)

    # NOTE: Some patients have multiple annotated images (eg. '010_HC.png' and '010_2HC.png'), so we
    # match on the 'HC.png' suffix rather than '_HC.png' and explicitly exclude the annotation images.
    image_paths = natsorted(
        p for p in glob(os.path.join(data_dir, "*HC.png")) if "Annotation" not in os.path.basename(p)
    )

    gt_dir = os.path.join(path, "masks")
    os.makedirs(gt_dir, exist_ok=True)

    gt_paths = []
    for image_path in tqdm(image_paths, desc="Rasterizing head circumference masks"):
        image_id = os.path.basename(image_path)[:-len(".png")]
        annotation_path = os.path.join(data_dir, f"{image_id}_Annotation.png")
        assert os.path.exists(annotation_path), f"The annotation for '{image_path}' is missing."

        gt_path = os.path.join(gt_dir, f"{image_id}.tif")
        if not os.path.exists(gt_path):
            _create_mask(annotation_path, gt_path)

        gt_paths.append(gt_path)

    assert len(image_paths) == len(gt_paths) == 999, len(image_paths)

    return image_paths, gt_paths


def get_hc18_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the HC18 dataset for segmentation of the fetal head in ultrasound images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_hc18_paths(path, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
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


def get_hc18_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the HC18 dataloader for segmentation of the fetal head in ultrasound images.

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
    dataset = get_hc18_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
