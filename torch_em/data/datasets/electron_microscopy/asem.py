import os

import numpy as np

import torch_em

from .. import util
from ... import ConcatDataset

try:
    import quilt3 as q3
    have_quilt = True
except ModuleNotFoundError:
    have_quilt = False


# The following volumes do not have labels:
#   - cell_8, cell_14, cell_15, cell_16, cell_17

# (E): "RuntimeError: Exception during zlib decompression: (-5)" (with `z5py`)
# (Y): similar shapes
# (N): dissimilar shapes (e.g. raw: (1000, 1100, 1200), labels: (200, 300, 400))

INCONSISTENT_VOLUMES = {
    "mito": ["cell_6.zarr", "cell_13.zarr", "cell_13a.zarr"],
    "golgi": ["cell_3.zarr", "cell_6.zarr"],
    "er": ["cell_3.zarr", "cell_6.zarr", "cell_13.zarr"],
}


VOLUMES = {
    "cell_1": "cell_1/cell_1.zarr",  # mito (Y) golgi (Y) er (Y)
    "cell_2": "cell_2/cell_2.zarr",  # mito (Y) golgi (Y) er (Y)
    "cell_3": "cell_3/cell_3.zarr",  # mito (Y) golgi (N) er (N)
    "cell_6": "cell_6/cell_6.zarr",  # mito (N) golgi (N) er (N)
    "cell_12": "cell_12/cell_12.zarr",  # ccp (Y)
    "cell_13": "cell_13/cell_13.zarr",  # ccp (Y) er (E) mito (N)
    "cell_13a": "cell_13a/cell_13a.zarr",  # np (Y) np_bottom (Y) mito (N)
}

ORGANELLES = {
    "mito": ["cell_1", "cell_2", "cell_3", "cell_6", "cell_13", "cell_13a"],
    "golgi": ["cell_1", "cell_2", "cell_3", "cell_6",],
    "er": ["cell_1", "cell_2", "cell_3", "cell_6",],
    "ccp": ["cell_12", "cell_13"],
    "np": ["cell_13a"],
    "np_bottom": ["cell_13a"]
}


def _download_asem_dataset(path, volume_ids, download):
    """https://open.quiltdata.com/b/asem-project"""
    if download and not have_quilt:
        raise ModuleNotFoundError("Please install quilt3: 'pip install quilt3'.")

    b = q3.Bucket("s3://asem-project")

    volume_paths = []
    for volume_id in volume_ids:
        volume_path = os.path.join(path, VOLUMES[volume_id])
        if not os.path.exists(volume_path):
            if not download:
                raise FileNotFoundError(f"{VOLUMES[volume_id]} is not found, and 'download' is set to False.")

            print(f"The ASEM dataset for sample '{volume_id}' is not available yet and will be downloaded and created.")
            print("Note that this dataset is large, so this step can take several hours (depending on your internet).")
            b.fetch(
                key=f"datasets/{VOLUMES[volume_id]}/volumes/labels/",
                path=os.path.join(volume_path, "volumes", "labels/")
            )
            b.fetch(
                key=f"datasets/{VOLUMES[volume_id]}/volumes/raw/",
                path=os.path.join(volume_path, "volumes", "raw/")
            )
            # let's get the group metadata keyfiles
            b.fetch(key=f"datasets/{VOLUMES[volume_id]}/.zgroup", path=f"{volume_path}/")
            b.fetch(key=f"datasets/{VOLUMES[volume_id]}/volumes/.zgroup", path=f"{volume_path}/volumes/")

        volume_paths.append(volume_path)

    return volume_paths


def _make_volumes_consistent(volume_path, organelle):
    import zarr

    have_inconsistent_volumes = False

    # we shouldn't load the volumes which are already consistent
    volume_name = os.path.split(volume_path)[-1]
    # there are organelles which aren't inconsistent at all, we ignore them.
    inc_vols = INCONSISTENT_VOLUMES.get(organelle)
    if inc_vols is None:  # i.e. the organelles have no inconsistency
        return have_inconsistent_volumes
    else:  # i.e. the organelles have some known inconsistency
        if volume_name not in inc_vols:  # if the current volume has inconsistency in the desired organelle or not
            return have_inconsistent_volumes

    with zarr.open(volume_path, "r+") as f:
        all_keys = list(f["volumes"].keys())
        # we shouldn't load the volume to make checks in case the processing has taken place already
        for this_key in all_keys:
            if this_key == f"raw_{organelle}":
                return True

        raw = f["volumes/raw"][:]

        this_key = f"volumes/labels/{organelle}"
        labels = f[this_key][:]

        if labels.shape != raw.shape:
            print("Found inconsistent volumes. Will save the desired crops of the volume.")
            have_inconsistent_volumes = True
            img_offset = np.array(
                np.array(f["volumes/raw"].attrs["offset"]) // np.array(f["volumes/raw"].attrs["resolution"])
            )
            label_offset = np.array(
                np.array(f[this_key].attrs["offset"]) // np.array(f[this_key].attrs["resolution"])
            )
            offset = label_offset - img_offset
            desired_slices = tuple(slice(o, s) for o, s in zip(offset, offset + labels.shape))
            new_raw = raw[desired_slices]

            assert new_raw.shape == labels.shape

            # HACK: current way-to-go is to create a new hierarchy where we store the desired volume patches
            # TODO: we want to integrate this so that this slicing can be done just by passing the offsets
            f.create_dataset(f"volumes/raw_{organelle}", data=new_raw, chunks=new_raw.shape)

    return have_inconsistent_volumes


def _check_input_args(input_arg, default_values):
    if input_arg is None:
        input_arg = default_values
    else:
        if isinstance(input_arg, str):
            assert input_arg in default_values
            input_arg = [input_arg]

    return input_arg


# TODO download the asem data and re-use this function in get_asem_data
def get_asem_data(path):
    pass


def get_asem_dataset(
    path, patch_shape, ndim, download, organelles=None, volume_ids=None, **kwargs
):
    """Dataset for the segmentation of organelles in FIB-SEM cells.

    This dataset provides access to 3d images of organelles (mitochondria, golgi, endoplasmic reticulum)
    segmented in cells. If you use this data in your research, please cite: https://doi.org/10.1083/jcb.202208005
    """
    # let's get the choice of organelles sorted
    organelles = _check_input_args(organelles, ORGANELLES)

    # now let's get the chosen volumes have the chosen organelles
    all_datasets = []
    for organelle in organelles:
        if volume_ids is None:
            volume_ids = ORGANELLES[organelle]
        else:
            if isinstance(volume_ids, str):
                volume_ids = [volume_ids]

            for volume_id in volume_ids:
                assert volume_id in ORGANELLES[organelle], \
                    f"The chosen volume and organelle combination does not match: '{volume_id}' & '{organelle}'"

        volume_paths = _download_asem_dataset(path, volume_ids, download)

        for volume_path in volume_paths:
            have_volumes_inconsistent = _make_volumes_consistent(volume_path, organelle)

            raw_key = f"volumes/raw_{organelle}" if have_volumes_inconsistent else "volumes/raw"
            dataset = torch_em.default_segmentation_dataset(
                volume_path, raw_key,
                volume_path, f"volumes/labels/{organelle}",
                patch_shape, ndim=ndim, is_seg_dataset=True,
                **kwargs
            )
            dataset.max_sampling_attempts = 5000
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
