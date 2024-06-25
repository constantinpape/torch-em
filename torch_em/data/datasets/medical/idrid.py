import os
from glob import glob
from pathlib import Path
from typing import Union, Tuple

import torch_em

from .. import util


TASKS = {
    "microaneurysms": r"1. Microaneurysms",
    "haemorrhages": r"2. Haemorrhages",
    "hard_exudates": r"3. Hard Exudates",
    "soft_exudates": r"4. Soft Exudates",
    "optic_disc": r"5. Optic Disc"
}


def get_idrid_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "data", "A.%20Segmentation")
    if os.path.exists(data_dir):
        return data_dir

    util.download_source_kaggle(
        path=path, dataset_name="aaryapatel98/indian-diabetic-retinopathy-image-dataset", download=download,
    )
    zip_path = os.path.join(path, "indian-diabetic-retinopathy-image-dataset.zip")
    util.unzip(zip_path=zip_path, dst=os.path.join(path, "data"))
    return data_dir


def _get_idrid_paths(path, split, task, download):
    data_dir = get_idrid_data(path=path, download=download)

    split = r"a. Training Set" if split == "train" else r"b. Testing Set"

    gt_paths = sorted(
        glob(
            os.path.join(data_dir, r"A. Segmentation", r"2. All Segmentation Groundtruths", split, TASKS[task], "*.tif")
        )
    )

    image_dir = os.path.join(data_dir, r"A. Segmentation", r"1. Original Images", split)
    image_paths = []
    for gt_path in gt_paths:
        gt_id = Path(gt_path).stem[:-3]
        image_path = os.path.join(image_dir, f"{gt_id}.jpg")
        image_paths.append(image_path)

    return image_paths, gt_paths


def get_idrid_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: str,
    task: str = "optic_disc",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """Dataset for segmentation of retinal lesions and optic disc in fundus images.

    The database is located at https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
    The dataloader makes use of an open-source version of the original dataset hosted on Kaggle.

    The dataset is from the IDRiD challenge:
    - https://idrid.grand-challenge.org/
    - Porwal et al. - https://doi.org/10.1016/j.media.2019.101561

    Please cite it if you use this dataset for a publication.
    """
    assert split in ["train", "test"]
    assert task in list(TASKS.keys())

    image_paths, gt_paths = _get_idrid_paths(path=path, split=split, task=task, download=download)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )

    return dataset


def get_idrid_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    split: str,
    task: str = "optic_disc",
    resize_inputs: bool = False,
    download: bool = False,
    **kwargs
):
    """
    Dataloader for segmentation of retinal lesions and optic disc in fundus images. See `get_idrid_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_idrid_dataset(
        path=path,
        patch_shape=patch_shape,
        split=split,
        task=task,
        resize_inputs=resize_inputs,
        download=download,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
