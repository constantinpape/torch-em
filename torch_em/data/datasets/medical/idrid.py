"""The IDRID dataset contains annotations for retinal lesions and optic disc segmentation
in Fundus images.

The database is located at https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
The dataloader makes use of an open-source version of the original dataset hosted on Kaggle.

The dataset is from the IDRiD challenge:
- https://idrid.grand-challenge.org/
- Porwal et al. - https://doi.org/10.1016/j.media.2019.101561
Please cite them if you use this dataset for your research.

The 'refined' version (selected with `version="refined"`) uses the same 81 IDRiD images, but replaces
the lesion annotations with the 'Refined IDRiD' release: expert-corrected and validated annotations for
the four original lesion types (microaneurysms, haemorrhages, hard exudates, soft / cotton-wool exudates),
plus three additional proliferative DR lesion types (neovascularization, vitreous haemorrhage, intraretinal
microvascular abnormalities) and anatomical context (optic disc, fovea, blood vessels, retinal region), all
merged into a single unified multi-class label mask per image. It is located at
https://doi.org/10.5281/zenodo.18676805 (CC BY 4.0). The `task` argument does not apply to this version, as
the label masks are not split by lesion type. Please cite the dataset if you use this version:
https://www.mdpi.com/2306-5729/11/2/30.
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

VERSIONS = ["v1", "refined"]

URL_REFINED = {
    "train": "https://zenodo.org/records/18676805/files/Train.tar",
    "test": "https://zenodo.org/records/18676805/files/Test.tar",
}
CHECKSUM_REFINED = {
    "train": "cb368cbfcdcbb2a9d22b95a99301aa4a06e2d7d510a2a151d488216f570136ba",
    "test": "3482c23e9c7a179960ea2a831ffe665bcbc825d497403ba5d0f386642eb4d821",
}


def get_idrid_data(
    path: Union[os.PathLike, str], download: bool = False, version: Literal["v1", "refined"] = "v1"
) -> str:
    """Download the IDRID dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.
        version: The version of the dataset. Either 'v1' (the original IDRiD lesion annotations) or
            'refined' (the 'Refined IDRiD' release with expert-corrected and additional lesion annotations).

    Returns:
        Filepath where the data is downloaded.
    """
    if version not in VERSIONS:
        raise ValueError(f"'{version}' is not a valid version. Please choose one of {VERSIONS}.")

    if version == "refined":
        data_dir = os.path.join(path, "refined")
        if os.path.exists(data_dir):
            return data_dir

        os.makedirs(data_dir, exist_ok=True)

        for split, url in URL_REFINED.items():
            tar_path = os.path.join(path, f"{split.capitalize()}.tar")
            util.download_source(path=tar_path, url=url, download=download, checksum=CHECKSUM_REFINED[split])
            util.unzip_tarfile(tar_path=tar_path, dst=data_dir, remove=False)

        return data_dir

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
    download: bool = False,
    version: Literal["v1", "refined"] = "v1",
) -> Tuple[List[str], List[str]]:
    """Get paths to the IDRID data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        task: The choice of labels for the specific task. Ignored when `version` is 'refined'.
        download: Whether to download the data if it is not present.
        version: The version of the dataset. Either 'v1' (the original IDRiD lesion annotations) or
            'refined' (the 'Refined IDRiD' release with expert-corrected and additional lesion annotations).

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    assert split in ["train", "test"]

    data_dir = get_idrid_data(path=path, download=download, version=version)

    if version == "refined":
        split_dir = "Train" if split == "train" else "Test"
        image_paths = sorted(glob(os.path.join(data_dir, split_dir, "Images", "*")))

        label_dir = os.path.join(data_dir, split_dir, "Labels")
        gt_paths = []
        for image_path in image_paths:
            stem = Path(image_path).stem
            label_path = os.path.join(label_dir, f"{stem}.png")
            if not os.path.exists(label_path):
                label_path = os.path.join(label_dir, f"{stem}_vessel.png")
            if not os.path.exists(label_path):
                raise RuntimeError(f"Could not find the matching label for the image at '{image_path}'.")
            gt_paths.append(label_path)

        return image_paths, gt_paths

    assert task in list(TASKS.keys())

    split = r"a. Training Set" if split == "train" else r"b. Testing Set"
    gt_paths = sorted(
        glob(
            os.path.join(data_dir, r"A. Segmentation", r"2. All Segmentation Groundtruths", split, TASKS[task], "*.tif")
        )
    )

    image_dir = os.path.join(data_dir, r"A. Segmentation", r"1. Original Images", split)
    image_paths = [os.path.join(image_dir, f"{Path(p).stem[:-3]}.jpg") for p in gt_paths]

    return image_paths, gt_paths


def get_idrid_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'],
    task: Literal['microaneurysms', 'haemorrhages', 'hard_exudates', 'soft_exudates', 'optic_disc'] = 'optic_disc',
    resize_inputs: bool = False,
    download: bool = False,
    version: Literal["v1", "refined"] = "v1",
    **kwargs
) -> Dataset:
    """Get the IDRID dataset for segmentation of retinal lesions and optic disc in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        task: The choice of labels for the specific task. Ignored when `version` is 'refined'.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        version: The version of the dataset. Either 'v1' (the original IDRiD lesion annotations) or
            'refined' (the 'Refined IDRiD' release with expert-corrected and additional lesion annotations).
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_idrid_paths(path, split, task, download, version)

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
    version: Literal["v1", "refined"] = "v1",
    **kwargs
) -> DataLoader:
    """Get the IDRID dataloader for segmentation of retinal lesions and optic disc in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        task: The choice of labels for the specific task. Ignored when `version` is 'refined'.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        version: The version of the dataset. Either 'v1' (the original IDRiD lesion annotations) or
            'refined' (the 'Refined IDRiD' release with expert-corrected and additional lesion annotations).
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_idrid_dataset(path, patch_shape, split, task, resize_inputs, download, version, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
