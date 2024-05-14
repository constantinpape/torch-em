import os
from glob import glob
from typing import Union, Tuple, Optional

import torch_em

from .. import util


URL = "https://zenodo.org/records/7960856/files/ISLES-2022.zip"
CHECKSUM = "f374895e383f725ddd280db41ef36ed975277c33de0e587a631ca7ea7ad45d6b"


def get_isles_data(path, download):
    os.makedirs(path, exist_ok=True)

    data_dir = os.path.join(path, "ISLES-2022")
    if os.path.exists(data_dir):
        return data_dir

    zip_path = os.path.join(path, "ISLES-2022.zip")
    util.download_source(path=zip_path, url=URL, download=download, checksum=CHECKSUM)
    util.unzip(zip_path=zip_path, dst=data_dir)

    return data_dir


def _get_isles_paths(path, modality, download):
    data_dir = get_isles_data(path=path, download=download)

    gt_paths = sorted(glob(os.path.join(data_dir, "derivatives", "sub-*", "**", "*.nii.gz"), recursive=True))

    dwi_paths = sorted(glob(os.path.join(data_dir, "sub-*", "**", "dwi", "*_dwi.nii.gz"), recursive=True))
    adc_paths = sorted(glob(os.path.join(data_dir, "sub-*", "**", "dwi", "*_adc.nii.gz"), recursive=True))

    if modality is None:
        image_paths = [(dwi_path, adc_path) for dwi_path, adc_path in zip(dwi_paths, adc_paths)]
    else:
        if modality == "dwi":
            image_paths = dwi_paths
        elif modality == "adc":
            image_paths = adc_paths
        else:
            raise ValueError

    return image_paths, gt_paths


def get_isles_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    modality: Optional[str] = None,
    download: bool = False,
    **kwargs
):
    """Dataset for ischemic stroke lesion segmentation in multimodal brain MRIs.

    The database is located at https://doi.org/10.5281/zenodo.7960856.
    This dataset is from the ISLES 2022 Challenge - https://doi.org/10.1038/s41597-022-01875-5.
    Please cite it if you use this dataset for a publication.
    """
    image_paths, gt_paths = _get_isles_paths(path=path, modality=modality, download=download)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key="data",
        label_paths=gt_paths,
        label_key="data",
        patch_shape=patch_shape,
        with_channels=modality is None,
        **kwargs
    )
    if "sampler" in kwargs:
        for ds in dataset.datasets:
            ds.max_sampling_attempts = 5000

    return dataset


def get_isles_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    modality: Optional[str] = None,
    download: bool = False,
    **kwargs
):
    """Dataloader for ischemic stroke lesion segmentation in multimodal brain MRIs. See `get_isles_dataset` for details.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_isles_dataset(path=path, patch_shape=patch_shape, modality=modality, download=download, **ds_kwargs)
    loader = torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
    return loader
