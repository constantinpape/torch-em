import os
from glob import glob
from pathlib import Path
from typing import Tuple, List, Optional, Union

import torch_em

from .. import util
from ....data import ConcatDataset


URL = {
    "braintumour": "https://drive.google.com/uc?export=download&id=1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU",
    "heart": "https://drive.google.com/uc?export=download&id=1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY",
    "liver": "https://drive.google.com/uc?export=download&id=1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu",
    "hippocampus": "https://drive.google.com/uc?export=download&id=1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C",
    "prostate": "https://drive.google.com/uc?export=download&id=1Ff7c21UksxyT4JfETjaarmuKEjdqe1-a",
    "lung": "https://drive.google.com/uc?export=download&id=1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi",
    "pancreas": "https://drive.google.com/uc?export=download&id=1YZQFSonulXuagMIfbJkZeTFJ6qEUuUxL",
    "hepaticvessel": "https://drive.google.com/uc?export=download&id=1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS",
    "spleen": "https://drive.google.com/uc?export=download&id=1jzeNU1EKnK81PyTsrx0ujfNl-t0Jo8uE",
    "colon": "https://drive.google.com/uc?export=download&id=1m7tMpE9qEcQGQjL_BdMD-Mvgmc44hG1Y",
}

CHECKSUM = {
    "braintumour": "",
    "heart": "",
    "liver": "",
    "hippocampus": "282d808a3e84e5a52f090d9dd4c0b0057b94a6bd51ad41569aef5ff303287771",
    "prostate": "",
    "lung": "",
    "pancreas": "",
    "hepaticvessel": "",
    "spleen": "",
    "colon": "",
}

FILENAMES = {
    "braintumour": "Task01_BrainTumour.tar",
    "heart": "Task02_Heart.tar",
    "liver": "Task03_Liver.tar",
    "hippocampus": "Task04_Hippocampus.tar",
    "prostate": "Task05_Prostate.tar",
    "lung": "Task06_Lung.tar",
    "pancreas": "Task07_Pancreas.tar",
    "hepaticvessel": "Task08_HepaticVessel.tar",
    "spleen": "Task09_Spleen.tar",
    "colon": "Task10_Colon.tar",
}


def _download_msd_data(path, task_name, download):
    os.makedirs(path, exist_ok=True)

    fpath = os.path.join(path, FILENAMES[task_name])

    util.download_source_gdrive(
        path=fpath, url=URL[task_name], download=download, checksum=None
    )
    try:
        util.unzip_tarfile(fpath, os.path.join(path, task_name), remove=False)
    except FileNotFoundError:
        print(
            f"{FILENAMES[task_name]} couldn't be downloaded automatically.",
            f"Please manually download the '{task_name}' dataset."
        )


def get_msd_dataset(
    path: str,
    patch_shape: Tuple[int, ...],
    ndim: int,
    task_names: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs
):
    """Dataset for semantic segmentation in 10 medical imaging datasets.
    Drive Link: https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2?usp=sharing

    This dataset is from the Medical Segmentation Decathlon Challenge.
    Link: http://medicaldecathlon.com/
    Please cite it if you use this dataset for a publication.

    Note: There is a possibility that the dataset cannot be downloaded from the drive links
    with the following errors (most issues are linked to `gdown`). Please download the dataset
    manually from the drive link to the MSD dataset (see above):
        - `Permission Denied`
        - `Cannot retrieve the public link of the file. You may need to change the permission
        to 'Anyone with the link', or have had many accesses.`
        - `Too many users have viewed or downloaded this file recently. Please try accessing
        the file again later. If the file you are trying to access is particularly large or is
        shared with many people, it may take up to 24 hours to be able to view or download the
        file. If you still can't access a file after 24 hours, contact your domain administrator.`


    Arguments:
        path: The path to prepare the dataset.
        patch_shape: The patch shape (for 2d or 3d patches)
        ndim: The dimensions of inputs (use `2` for getting `2d` patches, and `3` for getting 3d patches)
        task_names: The names for the 10 different segmentation tasks.
            - (default: None) If passed `None`, it takes all the tasks as inputs.
            - the names of the tasks are (see the challenge website for further details):
                - braintumour, heart, liver, hippocampus, prostate, lung, pancreas, hepaticvessel, spleen, colon
        download: Downloads the dataset
    """
    if task_names is None:
        task_names = list(URL.keys())
    else:
        if isinstance(task_names, str):
            task_names = [task_names]

    _datasets = []
    for task_name in task_names:
        _download_msd_data(path, task_name, download)
        image_paths = glob(os.path.join(path, task_name, Path(FILENAMES[task_name]).stem, "imagesTr", "*.nii.gz"))
        label_paths = glob(os.path.join(path, task_name, Path(FILENAMES[task_name]).stem, "labelsTr", "*.nii.gz"))
        this_dataset = torch_em.default_segmentation_dataset(
            image_paths, "data", label_paths, "data", patch_shape, ndim=ndim, **kwargs
        )
        _datasets.append(this_dataset)

    return ConcatDataset(*_datasets)


def get_msd_loader(
    path, patch_shape, batch_size, ndim, task_names=None, download=False, **kwargs
):
    """
    Dataloader for semantic segmentation from 10 highly variable medical segmentation tasks.
    See `get_msd_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_msd_dataset(path, patch_shape, ndim, task_names, download, **ds_kwargs)
    loader = torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
    return loader
