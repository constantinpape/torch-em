import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Tuple, List, Literal
import imageio.v3 as imageio
import numpy as np
import torch_em

from torch.utils.data import Dataset, DataLoader

from .. import util

def _extract_images(split: Literal["train", "test"], data_folder, output_dir):
    import h5py
    label_paths = natsorted(glob(os.path.join(data_folder, "*anno.bmp")))
    image_paths = [image_path for image_path in natsorted(glob(os.path.join(data_folder, "*.bmp")))
                   if image_path not in label_paths]
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    for image_file in tqdm(image_paths, desc=f"Extract images from {os.path.abspath(data_folder)}"):
        fname = os.path.basename(image_file).split(".")[0]
        if split not in fname:
            continue
        label_file = os.path.join(data_folder, f"{fname}_anno.bmp")
        assert os.path.exists(label_file), label_file

        image = imageio.imread(image_file)
        assert image.ndim == 3 and image.shape[-1] == 3

        segmentation = imageio.imread(label_file)
        assert image.shape[:-1] == segmentation.shape

        image = image.transpose((2, 0, 1))
        assert image.shape[1:] == segmentation.shape

        output_file = os.path.join(output_dir, split, f"{fname}.h5")
        with h5py.File(output_file, "a") as f:
            f.create_dataset("image", data=image, compression="gzip")
            f.create_dataset("labels/segmentation", data=segmentation, compression="gzip")

def get_glas_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the GlaS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded and preprocessed.
    """
    data_dir = os.path.join(path, "data", "Warwick_QU_Dataset")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(path=path, dataset_name="sani84/glasmiccai2015-gland-segmentation", download=download)
    util.unzip(zip_path=os.path.join(
        path, "glasmiccai2015-gland-segmentation.zip"), dst=os.path.join(path, "data"), remove=False
    )
    os.remove(os.path.join(path, "glasmiccai2015-gland-segmentation.zip"))
    return data_dir

def get_glas_paths(
    path: Union[os.PathLike], split: Literal["train", "val", "test"], download: bool = False
) -> List[str]:
    """Get paths to the GlaS data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The choice of data splits.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    data_dir = get_glas_data(path, download)
    if not os.path.exists(os.path.join(path, split)):
        _extract_images(split, data_dir, path)
    data_paths = natsorted(glob(os.path.join(path, split, "*.h5")))
    return data_paths

def get_glas_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the GlaS dataset for gland segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the input images.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    data_paths = get_glas_paths(path, split, download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=data_paths,
        raw_key="image",
        label_paths=data_paths,
        label_key="labels/segmentation",
        patch_shape=patch_shape,
        ndim=2,
        with_channels=True,
        **kwargs
    )


def get_glas_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "test"],
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the GlaS dataloader for gland segmentation.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_glas_dataset(path, patch_shape, split, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(ds, batch_size, **loader_kwargs)
