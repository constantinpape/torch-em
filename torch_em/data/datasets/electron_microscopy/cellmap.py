"""CellMap is a dataset for segmenting various organelles in electron microscopy.
It contains a large amount of annotation crops from several species.
This dataset is released for the `CellMap Segmentation Challenge`: https://cellmapchallenge.janelia.org/.
- Official documentation: https://janelia-cellmap.github.io/cellmap-segmentation-challenge/.
- Original GitHub repository for the toolbox: https://github.com/janelia-cellmap/cellmap-segmentation-challenge.
- And associated collection doi for the data: https://doi.org/10.25378/janelia.c.7456966.

Please cite them if you use this data for your research.
"""

import os
import time
from pathlib import Path
from threading import Lock
from typing import Union, Optional, Tuple, List, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd
from xarray import DataArray

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


def _download_cellmap_data(path, crops, resolution, padding=0, download=False):
    """Download scripts for the CellMap data.
    
    Inspired by https://github.com/janelia-cellmap/cellmap-segmentation-challenge/blob/main/src/cellmap_segmentation_challenge/cli/fetch_data.py

    NOTE: The download scripts below are intended to stay as close to the original `fetch-data` CLI,
    in order to ensure easy syncing with any changes to the original repository in future.
    """  # noqa

    # Importing packages locally.
    # NOTE: Keeping the relevant imports here to avoid `torch-em` throwing missing module error.

    try:
        from cellmap_segmentation_challenge.utils.fetch_data import read_group, subset_to_slice
        from cellmap_segmentation_challenge.utils.crops import fetch_crop_manifest, get_test_crops
    except ImportError:
        raise ModuleNotFoundError(
            "Please install 'cellmap_segmentation_challenge' package using "
            "'pip install git+https://github.com/janelia-cellmap/cellmap-segmentation-challenge.git'."
        )

    # NOTE: The imports below will come with the above lines of 'csc' installation.
    import structlog
    from xarray_ome_ngff import read_multiscale_group
    from xarray_ome_ngff.v04.multiscale import transforms_from_coords

    # Some important stuff.
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

    all_crops = []
    for crop in crops_parsed:
        log = log.bind(crop_id=crop.id, dataset=crop.dataset)

        # Get the crop id to a new list for forwarding them ahead.
        all_crops.append(crop.id)

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

        # Get the offset values for the ground truth crops.
        crop_multiscale_group = None
        for _, group in gt_source_group.groups():
            try:  # Get groups for all resolutions.
                crop_multiscale_group = read_multiscale_group(group, array_wrapper=array_wrapper)
                break
            except (ValueError, TypeError):
                continue

        if crop_multiscale_group is None:
            log.info(f"No multiscale groups found in '{crop.gt_source}'. No EM data can be fetched.")
            continue

        # Arrange the groups in descending order of resolution.
        gt_source_arrays_sorted = sorted(
            crop_multiscale_group.items(), key=lambda kv: np.prod(kv[1].shape), reverse=True
        )
        # Check whether the first resolution matches our expected resolution.
        if resolution != gt_source_arrays_sorted[0][0]:
            raise ValueError(
                f"The expected resolution '{resolution}' does not match the "
                f"highest available resolution '{gt_source_arrays_sorted[0][0]}'."
            )

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

        # Write all stuff in a crop-level h5 file.
        write_lock = Lock()
        with h5py.File(crop_path, "w") as f:
            # Store metadata
            f.attrs["crop_id"] = crop.id
            f.attrs["scale"] = scale.scale
            f.attrs["translation"] = translation.translation

            # Store inputs.
            f.create_dataset(name="raw_crop", data=em_crop, compression="gzip")

            def _fetch_and_write_label(label_name):
                gt_crop = gt_source_group[f"{label_name}/{resolution}"][:]
                with write_lock:
                    f.create_dataset(name=f"label_crop/{label_name}", data=gt_crop, compression="gzip")
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

    return path, all_crops


def get_cellmap_data(
    path: Union[os.PathLike, str],
    organelles: Optional[Union[str, List[str]]] = None,
    crops: Union[str, Sequence] = "all",
    resolution: str = "s0",
    download: bool = False,
) -> Tuple[str, List[str]]:
    """Downloads the CellMap training data.

    Args:
        path: Filepath to a folder where the data will be downloaded for further processing
        organelles: The choice of organelles to download. By default, loads all types of labels available.
            For one for multiple organelles, specify either like 'mito' or ['mito', 'cell'].
        crops: The choice of crops to download. By default, downloads `all` crops.
            For multiple crops, provide the crop ids as a sequence of crop ids.
        resolution: The choice of resolution. By default, downloads the highest resolution: `s0`.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is stored for further processing.
        List of crop ids.
    """

    data_path = os.path.join(path, "data_crops")
    os.makedirs(data_path, exist_ok=True)

    # Get the crops in 'csc' desired format.
    if isinstance(crops, Sequence) and not isinstance(crops, str):  # for multiple values
        crops = ",".join(str(c) for c in crops)

    # NOTE: The function below is comparable to the CLI `csc fetch-data` from the original repo.
    _data_path, final_crops = _download_cellmap_data(
        path=data_path,
        crops=crops,
        resolution=resolution,
        padding=0,
        download=download,
    )

    # Get the organelle-crop mapping.
    from cellmap_segmentation_challenge import utils

    # There is a file named 'train_crop_manifest' in the 'utils' sub-module. We need to get that first
    train_metadata_file = os.path.join(str(Path(utils.__file__).parent / "train_crop_manifest.csv"))
    train_metadata = pd.read_csv(train_metadata_file)

    # Let's get the label to crop mapping from the manifest file.
    organelle_to_crops = train_metadata.groupby('class_label')['crop_name'].apply(list).to_dict()

    # By default, 'organelles' set to 'None' will give you 'all' organelle types.
    if organelles is not None:  # The assumption here is that the user wants specific organelle(s).
        # Validate whether the organelle exists in the desired crops at all.
        if isinstance(organelles, str):
            organelles = [organelles]

        # Next, we check whether they match the crops.
        for curr_organelle in organelles:
            if curr_organelle not in organelle_to_crops:  # Check whether the organelle is valid or not.
                raise ValueError(f"The chosen organelle: '{curr_organelle}' seems to be an invalid choice.")

            # Lastly, we check whether the final crops have the organelle(s) or not.
            # Otherwise, we throw a warning and go ahead with the true valid choices.
            # NOTE: The priority below is higher for organelles than crops.
            for curr_crop in final_crops:
                if curr_crop not in organelle_to_crops.get(curr_organelle):
                    raise ValueError(f"The crop '{curr_crop}' does not have the chosen organelle '{curr_organelle}'.")

    if _data_path is None or len(_data_path) == 0:
        raise RuntimeError("Something went wrong. Please read the information logged above.")

    return data_path, final_crops


def get_cellmap_paths(
    path: Union[os.PathLike, str],
    organelles: Optional[Union[str, List[str]]] = None,
    crops: Union[str, Sequence] = "all",
    resolution: str = "s0",
    download: bool = False,
) -> List[str]:
    """Get the paths to CellMap training data.

    Args:
        path: Filepath to a folder where the data will be downloaded for further processing
        organelles: The choice of organelles to download. By default, loads all types of labels available.
            For one for multiple organelles, specify either like 'mito' or ['mito', 'cell'].
        crops: The choice of crops to download. By default, downloads `all` crops.
            For multiple crops, provide the crop ids as a sequence of crop ids.
        resolution: The choice of resolution. By default, downloads the highest resolution: `s0`.
        download: Whether to download the data if it is not present.

    Returns:
        List of the cropped volume data paths.
    """

    # Get the CellMap data crops.
    data_path, crops = get_cellmap_data(
        path=path, organelles=organelles, crops=crops, resolution=resolution, download=download
    )

    # Get all crops.
    volume_paths = [os.path.join(data_path, f"crop_{c}.h5") for c in crops]

    return volume_paths


def get_cellmap_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[str, ...],
    organelles: Optional[Union[str, List[str]]] = None,
    crops: Union[str, Sequence] = "all",
    resolution: str = "s0",
    download: bool = False,
    **kwargs,
) -> Dataset:
    """Get the dataset for the CellMap training data for organelle segmentation.

    Args:
        path: Filepath to a folder where the data will be downloaded for further processing.
        patch_shape: The patch shape to use for training.
        organelles: The choice of organelles to download. By default, loads all types of labels available.
            For one for multiple organelles, specify either like 'mito' or ['mito', 'cell'].
        crops: The choice of crops to download. By default, downloads `all` crops.
            For multiple crops, provide the crop ids as a sequence of crop ids.
        resolution: The choice of resolution. By default, downloads the highest resolution: `s0`.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_cellmap_paths(
        path=path, organelles=organelles, crops=crops, resolution=resolution, download=download
    )

    # Arrange the organelle choices as expected for loading labels.
    if organelles is None:
        organelles = "label_crop/all"
    else:
        if isinstance(organelles, str):
            organelles = f"label_crop/{organelles}"
        else:
            organelles = [f"label_crop/{curr_organelle}" for curr_organelle in organelles]
            kwargs = util.update_kwargs(kwargs, "with_label_channels", True)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw_crop",
        label_paths=volume_paths,
        label_key=organelles,
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_cellmap_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[str, ...],
    organelles: Optional[Union[str, List[str]]] = None,
    crops: Union[str, Sequence] = "all",
    resolution: str = "s0",
    download: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the dataloader for the CellMap training data for organelle segmentation.

    Args:
        path: Filepath to a folder where the data will be downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        organelles: The choice of organelles to download. By default, loads all types of labels available.
            For one for multiple organelles, specify either like 'mito' or ['mito', 'cell'].
        crops: The choice of crops to download. By default, downloads `all` crops.
            For multiple crops, provide the crop ids as a sequence of crop ids.
        resolution: The choice of resolution. By default, downloads the highest resolution: `s0`.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cellmap_dataset(path, patch_shape, organelles, crops, resolution, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
