import os

import torch_em

from . import util
from .. import ConcatDataset

try:
    import quilt3 as q3
    have_quilt = True
except ModuleNotFoundError:
    have_quilt = False


VOLUMES = {
    "cell_3": "cell_3/cell_3_tiff/",
    "cell_1": "cell_1/cell_1_tiff/",
    "cell_2": "cell_2/cell_2_tiff/",
    "cell_6": "cell_6/cell_6_tiff/",
    "cell_8": "cell_8/cell_8_tiff/",
    "cell_12": "cell_12/cell_12_tiff/",
    "cell_13": "cell_13/cell_13_tiff/",
    "cell_13a": "cell_13a/cell_13a_tiff/",
    "cell_14": "cell_14/cell_14_tiff/",
    "cell_15": "cell_15/cell_15_tiff/",
    "cell_16": "cell_16/cell_16_tiff/",
    "cell_17": "cell_17/cell_17_tiff/",
}

ORGANELLES = ["mito", "golgi", "er"]


def _download_asem_dataset(path, volume_ids, download):
    if download and not have_quilt:
        raise ModuleNotFoundError("Please install quilt3: 'pip install quilt3'.")

    b = q3.Bucket("s3://asem-project")

    volume_paths = []
    for volume_id in volume_ids:
        volume_path = os.path.join(path, VOLUMES[volume_id])
        if not os.path.exists(volume_path):
            if not download:
                raise FileNotFoundError(f"{VOLUMES[volume_id]} is not found, and 'download' is set to False.")

<<<<<<< HEAD
            print(f"The ASEM dataset for sample {volume_id} is not available yet and will be downloaded and created.")
            print("Note that this dataset is large, so this step can take several hours (depending on your internet).")
            b.fetch(key=f"datasets/{VOLUMES[volume_id]}", path=volume_path)

        volume_paths.append(volume_path)

        # import napari
        # import z5py

        # with z5py.File(volume_path, "r", use_zarr_format=True) as f:
        #     raw = f["volumes/raw_equalized_0.02"][:]
        #     er_labels = f["volumes/labels/er"][:]
        #     mito_labels = f["volumes/labels/mito"][:]
        #     golgi_labels = f["volumes/labels/golgi"][:]

        #     v = napari.Viewer()
        #     v.add_image(raw)
        #     v.add_labels(er_labels, visible=False)
        #     v.add_labels(mito_labels, visible=False)
        #     v.add_labels(golgi_labels, visible=False)
        #     napari.run()

        #     breakpoint()

    return volume_paths

=======
>>>>>>> 229e63f493fcbda37790f1ced3c34d22a10d157a

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
    segmented in cells. If you use this data in your research, please cite: https://doi.org/10.1083/jcb.202208005
    """
    volume_ids = _check_input_args(volume_ids, list(VOLUMES.keys()))
    organelles = _check_input_args(organelles, ORGANELLES)

    volume_paths = _download_asem_dataset(path, volume_ids, download)

    all_datasets = []
    for volume_path in volume_paths:
        print(volume_path)
        for organelle in organelles:
            dataset = torch_em.default_segmentation_dataset(
                volume_path, "volumes/raw_equalized_0.02",
                volume_path, f"volumes/labels/{organelle}",
                patch_shape, ndim=ndim, **kwargs
            )
            all_datasets.append(dataset)

    return ConcatDataset(*all_datasets)


def get_asem_loader(
    path, patch_shape, batch_size, ndim, download=False, organelles=None, volume_ids=None, **kwargs
):
    """Dataloader for organelle segmentation in FIB-SEM cells. See `get_asem_dataset` for details."""
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    ds = get_asem_dataset(path, patch_shape, ndim, download, organelles, volume_ids, **ds_kwargs)
    loader = torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)
    return loader
