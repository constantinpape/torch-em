"""
"""

import os
from pathlib import Path
from typing import Union, Sequence, Optional, Tuple, List

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def _get_scale_and_translation(label_zobj, resolution="s0"):
    all_scales = label_zobj.attrs.get("multiscales", [])[0]
    for ds in all_scales.get("datasets", []):
        if ds.get("path") == resolution:
            transforms = ds.get("coordinateTransformations", [])
            scale = next((t["scale"] for t in transforms if t["type"] == "scale"), None)
            translation = next((t["translation"] for t in transforms if t["type"] == "translation"), None)
            return scale, translation
    return None, None


def _download_cellmap_data(
    path: Union[os.PathLike, str],
    crops: str = "all",
    organelles: Sequence[str] = ("all"),
    resolution: str = "s0",
    download: bool = False,
):
    """Inspired by https://github.com/janelia-cellmap/cellmap-segmentation-challenge/blob/main/src/cellmap_segmentation_challenge/cli/fetch_data.py

    NOTE: The download scripts below are intended to stay as close to the original `fetch-data` CLI,
    in order to ensure easy syncing with any changes to the original repository in future.
    """  # noqa

    # Importing packages locally.
    # NOTE: Keeping the relevant imports here to avoid `torch-em` throwing missing module error.
    import time
    import structlog

    import h5py

    from xarray import DataArray
    from xarray_ome_ngff import read_multiscale_group
    from xarray_ome_ngff.v04.multiscale import transforms_from_coords

    from cellmap_segmentation_challenge.utils.fetch_data import read_group, subset_to_slice
    from cellmap_segmentation_challenge.utils.crops import fetch_crop_manifest, get_test_crops

    # Some important stuff.
    padding = 0
    fetch_save_start = time.time()
    log = structlog.get_logger()
    # Get the absolute path location to store crops.
    dest_path_abs = Path(path).absolute()
    dest_path_abs.mkdir(exist_ok=True)

    # Get the entire crop manifest.
    crops_from_manifest = fetch_crop_manifest()

    # Get the desired crop info from the manifest.
    if crops in ["all", "test"]:
        test_crops = get_test_crops()
        log.info(f"Found '{len(test_crops)}' test crops.")

    # Fetch all the crop manifests.
    if crops == "all":
        crops_parsed = crops_from_manifest + test_crops
    elif crops == "test":
        crops_parsed = test_crops
    else:  # Otherwise, custom crops are parsed.
        crops_split = tuple(int(x) for x in crops.split(","))
        crops_parsed = tuple(filter(lambda v: v.id in crops_split, crops_from_manifest))

    # Now get the crop ids.
    if len(crops_parsed) == 0:
        log.info(f"No crops found matching '{crops}'. Doing nothing.")
        return

    crop_ids = tuple(c.id for c in crops_parsed)
    log.info(f"Preparing to copy the following crops: '{crop_ids}'.")
    log.info(f"Data will be saved to '{dest_path_abs}'.")

    for crop in crops_parsed:
        log = log.bind(crop_id=crop.id, dataset=crop.dataset)

        # Check whether the crop path has been downloaded already or not.
        crop_path = dest_path_abs / f"crop_{crop.id}.h5"
        if crop_path.exists():
            log.info(f"The crop '{crop.id}' is already saved at '{crop_path}'.")
            log = log.unbind("crop_id", "dataset")
            continue

        # If 'download' is set to 'False', we do not go further from here.
        if not download:
            log.error(f"Cannot download the crop '{crop.id}' as 'download' is set to 'False'.")
            return

        # Get the EM volume of highest resolution.
        em_source_group = read_group(str(crop.em_url), storage_options={"anon": True})
        log.info(f"Found EM data at {crop.em_url}.")

        # Let's get the multiscale model of the source em group.
        array_wrapper = {"name": "dask_array", "config": {"chunks": "auto"}}
        em_source_arrays = read_multiscale_group(em_source_group, array_wrapper)
        em_s0_array = em_source_arrays[resolution]

        # Get the ground-truth (gt) masks.
        gt_source_group = read_group(str(crop.gt_source), storage_options={"anon": True})

        # Let's get all ground-truth hierarchies.
        # NOTE: Following the same as the original repo, relying on fs.find to avoid slowness in traversing online zarr.
        fs = gt_source_group.store.fs
        store_path = gt_source_group.store.path
        gt_files = fs.find(store_path)

        crop_group_inventory = tuple(fn.split(store_path)[-1] for fn in gt_files)
        crop_group_inventory = tuple(curr_cg[1:].split("/")[0] for curr_cg in crop_group_inventory)
        crop_group_inventory = np.unique(crop_group_inventory).tolist()
        crop_group_inventory = [curr_cg for curr_cg in crop_group_inventory if curr_cg not in [".zattrs", ".zgroup"]]

        gt_crop_shape = gt_source_group[f"all/{resolution}"].shape  # since "all" exists "al"ways, we rely on it.

        # scale, translation = _get_scale_and_translation(gt_source_group["all"])
        # if scale is None and translation is None:
        #     raise RuntimeError

        # Get the offset values for the ground truth crops.
        for _, group in gt_source_group.groups():
            try:
                crop_multiscale_group = read_multiscale_group(group, array_wrapper=array_wrapper)
                break
            except (ValueError, TypeError):
                continue

        if crop_multiscale_group is None:
            log.info(f"No multiscale groups found in '{crop.gt_source}'. No EM data can be fetched.")
            continue

        gt_source_arrays_sorted = sorted(
            crop_multiscale_group.items(), key=lambda kv: np.prod(kv[1].shape), reverse=True
        )
        # Check whether the first resolution matches our expected resolution.
        if resolution != gt_source_arrays_sorted[0][0]:
            raise ValueError

        _, (scale, translation) = transforms_from_coords(
            coords=gt_source_arrays_sorted[0][1].coords,  # accesing the metadata for 's0'.
            transform_precision=4,
        )

        # Compute the input reference crop from the ground truth metadata.
        starts = translation.translation
        stops = [start + size * vs for start, size, vs in zip(translation.translation, gt_crop_shape, scale.scale)]

        # Get the slices.
        coords = {dim: np.array([start, stop]) for dim, (start, stop) in zip(em_s0_array.dims, zip(starts, stops))}
        slices = subset_to_slice(em_s0_array, DataArray(dims=em_s0_array.dims, coords=coords))

        # Pad the slices (in voxel space)
        slices_padded = tuple(
            slice(max(0, sl.start - padding), min(sl.stop + padding, dim), sl.step)
            for sl, dim in zip(slices, em_s0_array.shape)
        )

        # Extract cropped EM volume from remote zarr files.
        em_crop = em_s0_array[tuple(slices_padded)].data.compute()

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock
        write_lock = Lock()

        # Write all stuff in a crop-level h5 file.
        with h5py.File(crop_path, "w") as f:
            # Store metadata
            f.attrs["crop_id"] = crop.id
            f.attrs["scale"] = scale.scale
            f.attrs["translation"] = translation.translation

            # Store inputs.
            f.create_dataset(name="raw_crop", shape=em_crop.shape, data=em_crop, compression="gzip")

            def _fetch_and_write_label(label_name):
                gt_crop = gt_source_group[f"{label_name}/{resolution}"][:]
                with write_lock:
                    f.create_dataset(
                        name=f"label_crop/{label_name}", shape=gt_crop.shape, data=gt_crop, compression="gzip",
                    )
                return label_name

            with ThreadPoolExecutor() as pool:
                futures = {pool.submit(_fetch_and_write_label, name): name for name in crop_group_inventory}
                for future in as_completed(futures):
                    label_name = future.result()
                    log.info(f"Saved ground truth crop '{crop.id}' for '{label_name}'.")

        log.info(f"Saved crop '{crop.id}' to '{crop_path}'.")
        log = log.unbind("crop_id", "dataset")

    log.info(f"Done after {time.time() - fetch_save_start:0.3f}s")
    log.info(f"Data saved to '{dest_path_abs}'.")


def get_cellmap_data(
    path: Union[os.PathLike, str],
    organelles: Sequence[str] = ("all"),
    crops: Sequence[str] = ("all"),
    resolution: str = "s0",
    download: bool = False,
) -> List:
    """Downloads the CellMap data.

    Args:
        path:
        organelles:
        crops:
        resolution:
        download:

    Returns:
        Filepath where the data is stored for further processing.
    """

    # NOTE: The function below is comparable to the CLI `csc fetch-data` from the original repo.
    _download_cellmap_data(path=path, crops=crops, organelles=organelles, resolution=resolution, download=download)


def get_cellmap_paths(
    path: Union[os.PathLike, str],
    organelles: Sequence[str] = ("all"),
    crops: Sequence[str] = ("all"),
    resolution: str = "s0",
    download: bool = False,
):
    """
    """

    # HACK: hard-coded atm to a certain crop for debugging purposes.
    # I need to do all changes to crop passing style here itself.
    crops = "234"

    # Get the CellMap data crops.
    get_cellmap_data(path=path, organelles=organelles, crops=crops, resolution=resolution, download=download)

    # TODO: Extend it to multiple crops later.
    volume_paths = os.path.join(path, f"crop_{crops}.h5")

    return [volume_paths]


def get_cellmap_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[str, ...],
    organelles: Sequence[str] = ("all"),
    crops: Sequence[str] = ("all"),
    resolution: str = "s0",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """
    """

    volume_paths = get_cellmap_paths(
        path=path, organelles=organelles, crops=crops, resolution=resolution, download=download
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw_crop",
        label_paths=volume_paths,
        label_key="label_crop/all",  # TODO: hard-coded atm.
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_cellmap_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[str, ...],
    organelles: Optional[Sequence[str]] = None,
    crops: Sequence[str] = ("all"),
    resolution: str = "s0",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cellmap_dataset(path, patch_shape, organelles, crops, resolution, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
