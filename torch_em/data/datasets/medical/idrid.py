"""The IDRID dataset contains annotations for retinal lesions and optic disc segmentation
in Fundus images.

The database is located at https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
The dataloader makes use of an open-source version of the original dataset hosted on Kaggle.

The dataset is from the IDRiD challenge:
- https://idrid.grand-challenge.org/
- Porwal et al. - https://doi.org/10.1016/j.media.2019.101561
Please cite them if you use this dataset for your research.
"""

import os
from glob import glob
from pathlib import Path
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


TASKS = {
    "microaneurysms": r"1. Microaneurysms",
    "haemorrhages": r"2. Haemorrhages",
    "hard_exudates": r"3. Hard Exudates",
    "soft_exudates": r"4. Soft Exudates",
    "optic_disc": r"5. Optic Disc"
}


def get_idrid_data(path: Union[os.PathLike, str], download: bool = False) -> str:
    """Download the IDRID dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data", "A.%20Segmentation")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    util.download_source_kaggle(
        path=path, dataset_name="aaryapatel98/indian-diabetic-retinopathy-image-dataset", download=download,
    )
    zip_path = os.path.join(path, "indian-diabetic-retinopathy-image-dataset.zip")
    util.unzip(zip_path=zip_path, dst=os.path.join(path, "data"))

    return data_dir


def get_idrid_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'test'],
    task: Literal['microaneurysms', 'haemorrhages', 'hard_exudates', 'soft_exudates', 'optic_disc'],
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths to the IDRID data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        task: The choice of labels for the specific task.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = get_idrid_data(path=path, download=download)

    assert split in ["train", "test"]
    assert task in list(TASKS.keys())

    split = r"a. Training Set" if split == "train" else r"b. Testing Set"
    gt_paths = sorted(
        glob(
            os.path.join(data_dir, r"A. Segmentation", r"2. All Segmentation Groundtruths", split, TASKS[task], "*.tif")
        )
    )

    image_dir = os.path.join(data_dir, r"A. Segmentation", r"1. Original Images", split)
    image_paths = [os.path.join(image_dir, f"{ Path(p).stem[:-3]}.jpg") for p in gt_paths]

    return image_paths, gt_paths


def get_idrid_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'],
    task: Literal['microaneurysms', 'haemorrhages', 'hard_exudates', 'soft_exudates', 'optic_disc'] = 'optic_disc',
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the IDRID dataset for segmentation of retinal lesions and optic disc in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        task: The choice of labels for the specific task.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_idrid_paths(path, split, task, download)

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


def get_idrid_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'],
    task: Literal['microaneurysms', 'haemorrhages', 'hard_exudates', 'soft_exudates', 'optic_disc'] = 'optic_disc',
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the IDRID dataloader for segmentation of retinal lesions and optic disc in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        task: The choice of labels for the specific task.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_idrid_dataset(path, patch_shape, split, task, resize_inputs, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
