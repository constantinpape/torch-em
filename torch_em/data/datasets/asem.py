import os

import torch_em

from . import util

try:
    import quilt3
    have_quilt = True
except ModuleNotFoundError:
    have_quilt = False


VOLUMES = {
    "cell_1": "cell_1/cell_1.zarr/",
    "cell_2": "cell_2/cell_2.zarr/",
    "cell_3": "cell_3/cell_3.zarr/",
    "cell_6": "cell_6/cell_6.zarr/",
    "cell_8": "cell_8/cell_8.zarr/",
    "cell_12": "cell_12/cell_12.zarr/",
    "cell_13": "cell_13/cell_13.zarr/",
    "cell_13a": "cell_13a/cell_13a.zarr/",
    "cell_14": "cell_14/cell_14.zarr/",
    "cell_15": "cell_15/cell_15.zarr/",
    "cell_16": "cell_16/cell_16.zarr/",
    "cell_17": "cell_17/cell_17.zarr/",
}

ORGANELLES = ["mito", "golgi", "er"]


def _download_asem_dataset(path, volume_id, download):
    volume_path = os.path.join(path, VOLUMES[volume_id])
    if not os.path.exists(volume_path):
        if not download:
            raise FileNotFoundError(f"{VOLUMES[volume_id]} is not found, and 'download' is set to False.")

        if not have_quilt and download:
            print("Please install quilt3: 'pip install quilt3'.")
            quit()

        b = quilt3.Bucket("s3://asem-project")
        b.fetch(key=f"datasets/{VOLUMES[volume_id]}", path=volume_path)


def _check_input_args(input_arg, default_values):
    if input_arg is None:
        input_arg = default_values
    else:
        if isinstance(input_arg, str):
            assert input_arg in default_values
            input_arg = [input_arg]

    return input_arg


def get_asem_dataset(
    path, patch_shape, ndim, download, organelles=None, volume_ids=None, **kwargs
):
    """Dataset for the segmentation of organelles in FIB-SEM cells.

    This dataset provides access to 3d images of organelles (mitochondria, golgi, endoplasmic reticulum)
    segmented in cells. If you use this data in your research, please cite
    https://doi.org/10.1083/jcb.202208005
    """
    volume_ids = _check_input_args(volume_ids, list(VOLUMES.keys()))
    organelles = _check_input_args(organelles, ORGANELLES)

    for volume_id in volume_ids:
        _download_asem_dataset(path, volume_id, download)


def get_asem_loader(
    path, patch_shape, batch_size, ndim, download=False, organelles=None, volume_ids=None, **kwargs
):
    """TODO: description of the loader"""
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_asem_dataset(path, patch_shape, ndim, download, volume_ids, **ds_kwargs)
    loader = torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
    return loader
