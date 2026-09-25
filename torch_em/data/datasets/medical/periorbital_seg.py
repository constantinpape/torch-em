"""The Periorbital Segmentation dataset contains annotations for periorbital anatomical
structures (eyebrow, sclera, iris, caruncle, eyelid) in cropped eye images.

The dataset bundles both the cropped eye images and the segmentation masks (no need to
source images separately from the Chicago Face Database or CelebAMask-HQ). It combines two
subsets:
- 'celeb': 2015 image-mask pairs cropped from CelebAMask-HQ.
- 'cfd': 827 image-mask pairs cropped from the Chicago Face Database.
(The archive also has a 'combined_final_data' folder, but it is inconsistent - it has 3670
images and only 2842 masks - so this module sources images and masks from the per-origin
'celeb_final_data' and 'cfd_final_data' folders instead, which are fully paired.)

The label legend (confirmed from the dataset's own 'coco2voc_aux.py' preprocessing script) is:
- background: 0, eyebrow: 1, sclera: 2, iris (incl. pupil): 3, caruncle: 4, eyelid: 5

The dataset is located at https://doi.org/10.5281/zenodo.13916845 (Zenodo, CC BY 4.0).
This dataset is from the publication https://doi.org/10.48550/arXiv.2409.20407.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URL = "https://zenodo.org/records/13916845/files/periorbital_dataset.zip?download=1"
CHECKSUM = "7c1cd92e4bc58e5b288c6eeb1769a98fa119338bfb25a2d3bb7ccebed8d99011"

SUBSETS = ["celeb", "cfd"]


def get_periorbital_seg_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the Periorbital Segmentation dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored.
    """
    data_dir = os.path.join(path, "periorbital_dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, "periorbital_dataset.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=path)

    return data_dir


def get_periorbital_seg_paths(
    path: Union[os.PathLike, str],
    subset: Union[Literal["celeb", "cfd"], List[str]] = SUBSETS,
    download: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get paths to the Periorbital Segmentation data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        subset: The choice of data subset(s). Either 'celeb', 'cfd', or a list of both. By default, loads both.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_periorbital_seg_data(path, download)

    subsets = [subset] if isinstance(subset, str) else subset
    for s in subsets:
        if s not in SUBSETS:
            raise ValueError(f"'{s}' is not a valid subset. Please choose from {SUBSETS}.")

    image_paths, gt_paths = [], []
    for s in subsets:
        this_image_paths = natsorted(glob(os.path.join(data_dir, f"{s}_final_data", f"{s}_output_images", "*.jpg")))
        this_gt_dir = os.path.join(data_dir, f"{s}_final_data", f"{s}_output_masks")
        for image_path in this_image_paths:
            fname = os.path.splitext(os.path.basename(image_path))[0]
            gt_path = os.path.join(this_gt_dir, f"{fname}.png")
            if not os.path.exists(gt_path):
                continue
            image_paths.append(image_path)
            gt_paths.append(gt_path)

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0, (
        "No image-mask pairs were found. The expected per-subset 'celeb_output_images' / 'celeb_output_masks' "
        f"(and 'cfd_output_images' / 'cfd_output_masks') layout may not match the actual structure of the "
        f"downloaded data. Please inspect the data at '{data_dir}' and update the search pattern accordingly."
    )

    return image_paths, gt_paths


def get_periorbital_seg_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    subset: Union[Literal["celeb", "cfd"], List[str]] = SUBSETS,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the Periorbital Segmentation dataset for periorbital anatomical structure segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        subset: The choice of data subset(s). Either 'celeb', 'cfd', or a list of both. By default, loads both.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_periorbital_seg_paths(path, subset, download)

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
        is_seg_dataset=False,
        patch_shape=patch_shape,
        **kwargs
    )


def get_periorbital_seg_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    subset: Union[Literal["celeb", "cfd"], List[str]] = SUBSETS,
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the Periorbital Segmentation dataloader for periorbital anatomical structure segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        subset: The choice of data subset(s). Either 'celeb', 'cfd', or a list of both. By default, loads both.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_periorbital_seg_dataset(path, patch_shape, subset, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
