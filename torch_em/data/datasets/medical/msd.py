"""The MSD dataset contains annotations for 10 different datasets,
composed of multiple structures / organs across various medical imaging modalities.

Here's an example for how to pass different tasks:
```python
# we want to get datasets for one task, eg. "heart"
task_names = ["heart"]

# Example: We want to get datasets for multiple tasks
# NOTE 1: it's important to note that datasets with similar number of modality (channels) can be paired together.
# to use different datasets together, you need to use "raw_transform" to update inputs per dataset
# to pair as desired patch shapes per batch.
# Example 1: "heart", "liver", "lung" all have one modality inputs
task_names = ["heart", "lung", "liver"]

# Example 2: "braintumour" and "prostate" have multi-modal inputs, however the no. of modalities are not equal.
# hence, you can use only one at a time.
task_names = ["prostate"]
```

This dataset is from the Medical Segmentation Decathlon Challenge:
- Antonelli et al. - https://doi.org/10.1038/s41467-022-30695-9
- Link - http://medicaldecathlon.com/

Please cite them if you use this dataset for your research.
"""

import os
from glob import glob
from pathlib import Path
from typing import Tuple, List, Union

from torch.utils.data import Dataset, DataLoader

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
    "liver": "4007d9db1acda850d57a6ceb2b3998b7a0d43f8ad5a3f740dc38bc0cb8b7a2c5",
    "hippocampus": "282d808a3e84e5a52f090d9dd4c0b0057b94a6bd51ad41569aef5ff303287771",
    "prostate": "8cbbd7147691109b880ff8774eb6ab26704b1be0935482e7996a36a4ed31ec79",
    "lung": "f782cd09da9cf7a3128475d4a53650d371db10f0427aa76e166fccfcb2654161",
    "pancreas": "e40181a0229ca85c2588d6ebb90fa6674f84eb1e66f0f968cda088d011769732",
    "hepaticvessel": "ee880799f12e3b6e1ef2f8645f6626c5b39de77a4f1eae6f496c25fbf306ba04",
    "spleen": "dfeba347daae4fb08c38f4d243ab606b28b91b206ffc445ec55c35489fa65e60",
    "colon": "a26bfd23faf2de703f5a51a262cd4e2b9774c47e7fb86f0e0a854f8446ec2325",
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


def get_msd_data(path: Union[os.PathLike, str], task_name: str, download: bool = False) -> str:
    """Download the MSD dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        task_name: The choice of specific task.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downloaded.
    """
    data_dir = os.path.join(path, "data", task_name)
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(path, exist_ok=True)

    fpath = os.path.join(path, FILENAMES[task_name])
    util.download_source(path=fpath, url=URL[task_name], download=download, checksum=None)
    util.unzip_tarfile(tar_path=fpath, dst=data_dir, remove=False)

    return data_dir


def get_msd_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    task_names: Union[str, List[str]],
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the MSD dataset for semantic segmentation in medical imaging datasets.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        task_names: The names for the 10 different segmentation tasks (see the challenge website for further details):
            1. tasks with 1 modality inputs are: heart, liver, hippocampus, lung, pancreas, hepaticvessel, spleen, colon
            2. tasks with multi-modality inputs are:
                - braintumour: with 4 modality (channel) inputs
                - prostate: with 2 modality (channel) inputs
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    if isinstance(task_names, str):
        task_names = [task_names]

    _datasets = []
    for task_name in task_names:
        data_dir = get_msd_data(path, task_name, download)
        image_paths = glob(os.path.join(data_dir, Path(FILENAMES[task_name]).stem, "imagesTr", "*.nii.gz"))
        label_paths = glob(os.path.join(data_dir, Path(FILENAMES[task_name]).stem, "labelsTr", "*.nii.gz"))

        if task_name in ["braintumour", "prostate"]:
            kwargs["with_channels"] = True

        this_dataset = torch_em.default_segmentation_dataset(
            raw_paths=image_paths,
            raw_key="data",
            label_paths=label_paths,
            label_key="data",
            patch_shape=patch_shape,
            **kwargs
        )
        _datasets.append(this_dataset)

    return ConcatDataset(*_datasets)


def get_msd_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    task_names: Union[str, List[str]],
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the MSD dataloader for semantic segmentation in medical imaging datasets.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        task_names: The names for the 10 different segmentation tasks (see the challenge website for further details):
            1. tasks with 1 modality inputs are: heart, liver, hippocampus, lung, pancreas, hepaticvessel, spleen, colon
            2. tasks with multi-modality inputs are:
                - braintumour: with 4 modality (channel) inputs
                - prostate: with 2 modality (channel) inputs
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_msd_dataset(path, patch_shape, task_names, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
