"""The UWaterloo Skin dataset contains annotations for skin lesion segmentation in dermoscopy images.

The database is located at
https://uwaterloo.ca/vision-image-processing-lab/research-demos/skin-cancer-detection.

Please cite it if you use this dataset for a publication.
"""

import os
import shutil
from glob import glob
from urllib.parse import urljoin
from urllib3.exceptions import ProtocolError
from typing import Tuple, Union, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


BASE_URL = "https://uwaterloo.ca/vision-image-processing-lab/sites/ca.vision-image-processing-lab/files/uploads/files/"


ZIPFILES = {
    "set1": "skin_image_data_set-1.zip",  # patients with melanoma
    "set2": "skin_image_data_set-2.zip"  # patients without melanoma
}

CHECKSUMS = {
    "set1": "1788cd3eb7a4744012aad9a154e514fc5b82b9f3b19e31cc1b6ded5fc6bed297",
    "set2": "108a818baf20b36ef4544ebda10a8075dad99e335f0535c9533bb14cb02b5c53"
}


def get_uwaterloo_skin_data(
    path: Union[os.PathLike, str], chosen_set: Literal["set1", "set2"], download: bool = False
) -> str:
    """Download the UWaterloo Skin dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        chosen_set: The choice of data subset.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    assert chosen_set in ZIPFILES.keys(), f"'{chosen_set}' is not a valid set."

    data_dir = os.path.join(path, f"{chosen_set}_Data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    zip_path = os.path.join(path, ZIPFILES[chosen_set])
    url = urljoin(BASE_URL, ZIPFILES[chosen_set])

    try:
        util.download_source(path=zip_path, url=url, download=download, checksum=CHECKSUMS[chosen_set])
    except ProtocolError:  # the 'uwaterloo.ca' quite randomly times out of connections, pretty weird.
        msg = "The server seems to be unreachable at the moment. "
        msg += f"We recommend downloading the data manually, from '{url}' at '{path}'. "
        print(msg)
        quit()

    util.unzip(zip_path=zip_path, dst=path)

    setnum = chosen_set[-1]
    tmp_dir = os.path.join(path, fr"Skin Image Data Set-{setnum}")
    shutil.move(src=tmp_dir, dst=data_dir)

    return data_dir


def get_uwaterloo_skin_paths(path: Union[os.PathLike, str], download: bool = False) -> Tuple[List[str], List[str]]:
    """Get paths to the UWaterloo Skin data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_uwaterloo_skin_data(path, "set1", download)
    image_paths = sorted(glob(os.path.join(data_dir, "skin_data", "melanoma", "*", "*_orig.jpg")))
    gt_paths = sorted(glob(os.path.join(data_dir, "skin_data", "melanoma", "*", "*_contour.png")))

    data_dir = get_uwaterloo_skin_data(path, "set2", download)
    image_paths.extend(sorted(glob(os.path.join(data_dir, "skin_data", "notmelanoma", "*", "*_orig.jpg"))))
    gt_paths.extend(sorted(glob(os.path.join(data_dir, "skin_data", "notmelanoma", "*", "*_contour.png"))))

    return image_paths, gt_paths


def get_uwaterloo_skin_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the UWaterloo Skin dataset for skin lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_uwaterloo_skin_paths(path, download)

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


def get_uwaterloo_skin_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the UWaterloo Skin dataloader for skin lesion segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        resize_inputs: Whether to resize inputs to the desired patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_uwaterloo_skin_dataset(path, patch_shape, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
