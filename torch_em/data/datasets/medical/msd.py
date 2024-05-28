import os
from glob import glob
from pathlib import Path
from typing import Tuple, List, Optional, Union

import torch_em

from .. import util
from ....data import ConcatDataset


URL = {
    "braintumour": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
    "heart": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar",
    "liver": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar",
    "hippocampus": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar",
    "prostate": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar",
    "lung": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar",
    "pancreas": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar",
    "hepaticvessel": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar",
    "spleen": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar",
    "colon": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar",
}

CHECKSUM = {
    "braintumour": "d423911308d2ae5396d9c6bf4fad2b68cfde2dd09044269da9c0d639c22753c4",
    "heart": "4277dc6dfe100142aa8060e895f6ff0f81c5b733703ea250bd294df8f820bcba",
    "liver": "",
    "hippocampus": "282d808a3e84e5a52f090d9dd4c0b0057b94a6bd51ad41569aef5ff303287771",
    "prostate": "8cbbd7147691109b880ff8774eb6ab26704b1be0935482e7996a36a4ed31ec79",
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


def get_msd_data(path, task_name, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "data", task_name)
    if os.path.exists(data_dir):
        return data_dir

    fpath = os.path.join(path, FILENAMES[task_name])

    util.download_source(path=fpath, url=URL[task_name], download=download, checksum=None)
    util.unzip_tarfile(tar_path=fpath, dst=data_dir, remove=False)

    return data_dir


def get_msd_dataset(
    path: str,
    patch_shape: Tuple[int, ...],
    ndim: int,
    task_names: Optional[Union[str, List[str]]] = None,
    download: bool = False,
    **kwargs
):
    """Dataset for semantic segmentation in 10 medical imaging datasets.

    This dataset is from the Medical Segmentation Decathlon Challenge:
    - Antonelli et al. - https://doi.org/10.1038/s41467-022-30695-9
    - Link - http://medicaldecathlon.com/

    Please cite it if you use this dataset for a publication.

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
        data_dir = get_msd_data(path, task_name, download)
        image_paths = glob(os.path.join(data_dir, Path(FILENAMES[task_name]).stem, "imagesTr", "*.nii.gz"))
        label_paths = glob(os.path.join(data_dir, Path(FILENAMES[task_name]).stem, "labelsTr", "*.nii.gz"))

        # for image_path, label_path in zip(image_paths, label_paths):
        #     import nibabel as nib

        #     image, gt = nib.load(image_path), nib.load(label_path)
        #     image, gt = image.get_fdata(), gt.get_fdata()

        #     print(image.shape, gt.shape)
        #     breakpoint()

        #     image = image.transpose(2, 0, 1)
        #     gt = gt.transpose(2, 0, 1)

        #     import napari
        #     v = napari.Viewer()
        #     v.add_image(image)
        #     v.add_labels(gt.astype("uint8"))
        #     napari.run()

        this_dataset = torch_em.default_segmentation_dataset(
            raw_paths=image_paths,
            raw_key="data",
            label_paths=label_paths,
            label_key="data",
            patch_shape=patch_shape,
            ndim=ndim,
            **kwargs
        )
        _datasets.append(this_dataset)

    return ConcatDataset(*_datasets)


def get_msd_loader(
    path, patch_shape, batch_size, ndim, task_names=None, download=False, **kwargs
):
    """Dataloader for semantic segmentation from 10 highly variable medical segmentation tasks.
    See `get_msd_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    ds = get_msd_dataset(path, patch_shape, ndim, task_names, download, **ds_kwargs)
    loader = torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
    return loader
