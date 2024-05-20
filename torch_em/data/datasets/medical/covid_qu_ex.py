import os
from glob import glob
from natsort import natsorted
from typing import Tuple, Union, Optional

import torch_em

from .. import util


def get_covid_qu_ex_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "data")
    if os.path.exists(data_dir):
        return data_dir

    util.download_source_kaggle(path=path, dataset_name="anasmohammedtahir/covidqu", download=download)
    zip_path = os.path.join(path, "covidqu.zip")
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def _get_covid_qu_ex_paths(path, split, task, patient_type, segmentation_mask, download):
    data_dir = get_covid_qu_ex_data(path=path, download=download)

    if task == "lung":
        task = r"Infection Segmentation Data/Infection Segmentation Data"
    elif task == "infection":
        task = r"Lung Segmentation Data/Lung Segmentation Data"
    else:
        raise ValueError(f"'{task}' is not a valid task.")

    if patient_type == "covid19":
        patient_type = "COVID-19"
    elif patient_type == "non-covid":
        patient_type = "Non-COVID"
    elif patient_type == "normal":
        patient_type = "Normal"
    else:
        if patient_type is None:
            patient_type = "*"
        else:
            raise ValueError(f"'{patient_type}' is not a valid patient type.")

    base_dir = os.path.join(data_dir, task, split.title(), patient_type)

    if segmentation_mask == "lung":
        segmentation_mask = r"lung masks"
    elif segmentation_mask == "infection":
        segmentation_mask = r"infection masks"
    else:
        if segmentation_mask is None:
            segmentation_mask = "*"
        else:
            raise ValueError(f"'{segmentation_mask}' is not a valid segmentation task.")

    image_paths = natsorted(glob(os.path.join(base_dir, "images", "*")))
    gt_paths = natsorted(glob(os.path.join(base_dir, segmentation_mask, "*")))

    print(len(image_paths), len(gt_paths))

    breakpoint()

    return image_paths, gt_paths


# TODO: simplify the data logic here, there are way too many choices for the users to make here
# reference: this is actually a pretty huge (and diverse) dataset.
def get_covid_qu_ex_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    task: str,
    split: str,
    patient_type: Optional[str] = None,
    segmentation_mask: Optional[str] = None,
    download: bool = False,
    **kwargs
):
    """Dataset for infection and lung segmentation in chest x-ray images.

    The database is located at https://www.kaggle.com/datasets/anasmohammedtahir/covidqu

    The dataset comes from Rahman et al. - https://doi.org/10.1016/j.compbiomed.2021.104319
    Please cite it if you use this dataset for a publication.
    """
    assert split.lower() in ["train", "val", "test"], f"'{split}' is not a valid split."

    if segmentation_mask is not None:
        assert segmentation_mask in ["infection", "lung"]

    image_paths, gt_paths = _get_covid_qu_ex_paths(
        path=path,
        split=split,
        task=task,
        patient_type=patient_type,
        segmentation_mask=segmentation_mask,
        download=download,
    )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        **kwargs
    )

    return dataset


def get_covid_qu_ex_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    task: str,
    split: str,
    patient_type: Optional[str] = None,
    segmentation_mask: Optional[str] = None,
    download: bool = False,
    **kwargs
):
    """Dataloader for infection and lung segmentation in CXR images. See `get_covid_qu_ex_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_covid_qu_ex_dataset(
        path=path,
        patch_shape=patch_shape,
        task=task,
        split=split,
        patient_type=patient_type,
        segmentation_mask=segmentation_mask,
        download=download,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
