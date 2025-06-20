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

from elf.io import open_file

from .. import util


def _download_cellmap_data(path, crops, resolution, padding, download=False):
    """Download scripts for the CellMap data.
    
    Inspired by https://github.com/janelia-cellmap/cellmap-segmentation-challenge/blob/main/src/cellmap_segmentation_challenge/cli/fetch_data.py

    NOTE: The download scripts below are intended to stay as close to the original `fetch-data` CLI,
    in order to ensure easy syncing with any changes to the original repository in future.
    """  # noqa

    # Importing packages locally.
    # NOTE: Keeping the relevant imports here to avoid `torch-em` throwing missing module error.

    try:
        from cellmap_segmentation_challenge.utils.fetch_data import read_group, subset_to_slice
        from cellmap_segmentation_challenge.utils.crops import fetch_crop_manifest, get_test_crops, TestCropRow
    except ImportError:
        raise ModuleNotFoundError(
            "Please install 'cellmap_segmentation_challenge' package using "
            "'pip install git+https://github.com/janelia-cellmap/cellmap-segmentation-challenge.git'."
        )

    # The imports below will come with the above lines of 'csc' installation.
    import structlog
    from xarray_ome_ngff import read_multiscale_group
    from xarray_ome_ngff.v04.multiscale import transforms_from_coords

    # Some important stuff.
    fetch_save_start = time.time()
    log = structlog.get_logger()
    array_wrapper = {"name": "dask_array", "config": {"chunks": "auto"}}

    # Get the absolute path location to store crops.
    dest_path_abs = Path(path).absolute()
    dest_path_abs.mkdir(exist_ok=True)

    # Get the entire crop manifest.
    crops_from_manifest = fetch_crop_manifest()

    # Get the desired crop info from the manifest.
    if crops == "all":
        crops_parsed = crops_from_manifest
    elif crops == "test":
        crops_parsed = get_test_crops()
        log.info(f"Found '{len(crops_parsed)}' test crops.")
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

        # Check whether the crop is a part of the test crops, i.e. where GT masks is not available.
        if isinstance(crop.gt_source, TestCropRow):
            log.info(f"The test crop '{crop.id}' does not have GT data. Fetching em data only.")
        else:
            log.info(f"Fetching GT data for crop '{crop.id}' from '{crop.gt_source}'.")

            # Get the ground-truth (gt) masks.
            gt_source_group = read_group(str(crop.gt_source), storage_options={"anon": True})

            log.info(f"Found GT data at '{crop.gt_source}'.")

            # Let's get all ground-truth hierarchies.
            # NOTE: Following same as the original repo, relying on fs.find to avoid slowness in traversing online zarr.
            fs = gt_source_group.store.fs
            store_path = gt_source_group.store.path
            gt_files = fs.find(store_path)

            crop_group_inventory = tuple(fn.split(store_path)[-1] for fn in gt_files)
            crop_group_inventory = tuple(curr_cg[1:].split("/")[0] for curr_cg in crop_group_inventory)
            crop_group_inventory = np.unique(crop_group_inventory).tolist()
            crop_group_inventory = [
                curr_cg for curr_cg in crop_group_inventory if curr_cg not in [".zattrs", ".zgroup"]
            ]

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

        # Get the EM volume group.
        em_source_group = read_group(str(crop.em_url), storage_options={"anon": True})
        log.info(f"Found EM data at '{crop.em_url}'.")

        # Let's get the multiscale model of the source em group.
        em_source_arrays = read_multiscale_group(em_source_group, array_wrapper)

        # Next, we need to rely on the scales of each resolution to identify whether the resolution-level is same
        # for the EM volume and corresponding ground-truth mask crops (if available).

        # For this, we first extract the EM volume scales per resolution.
        em_resolutions = {}
        for res_key, array in em_source_arrays.items():
            try:
                _, (em_scale, em_translation) = transforms_from_coords(array.coords, transform_precision=4)
                em_resolutions[res_key] = (em_scale.scale, em_translation.translation)
            except Exception:
                continue

        if isinstance(crop.gt_source, TestCropRow):
            # Choose the scale ratio threshold (from the original scripts)
            ratio_threshold = 0.8  # NOTE: hard-coded atm to follow along the original data download code logic.

            # Choose the matching resolution level with marked GT.
            em_level = next(
                (
                    k for k, (scale, _) in em_resolutions.items()
                    if all(s / vs > ratio_threshold for s, vs in zip(scale, crop.gt_source.voxel_size))
                ), None
            )

            assert em_level is not None, "There has to be a scale match for the EM volume. Something went wrong."

            scale = em_resolutions[em_level][0]
            em_array = em_source_arrays[em_level]

            # Get the slices (NOTE: there is info for some crop logic stuff)
            starts = crop.gt_source.translation
            stops = tuple(
                start + size * vs for start, size, vs in zip(starts, crop.gt_source.shape, crop.gt_source.voxel_size)
            )
            coords = em_array.coords.copy()
            for k, v in zip(em_array.coords.keys(), np.array((starts, stops)).T):
                coords[k] = v

            slices = subset_to_slice(outer_array=em_array, inner_array=DataArray(dims=em_array.dims, coords=coords))

            # Set 'gt_level' to 'None' for better handling of crops without labels.
            gt_level = None

        else:
            # Next, we extract the ground-truth scales per resolution (for labeled crops).
            gt_resolutions = {}
            for res_key, array in crop_multiscale_group.items():
                try:
                    _, (gt_scale, gt_translation) = transforms_from_coords(array.coords, transform_precision=4)
                    gt_resolutions[res_key] = (gt_scale.scale, gt_translation.translation)
                except Exception:
                    continue

            # Now, we find the matching scales and use the respoective "resolution" keys.
            matching_keys = []
            for gt_key, (gt_scale, gt_translation) in gt_resolutions.items():
                for em_key, (em_scale, em_translation) in em_resolutions.items():
                    if np.allclose(gt_scale, em_scale, rtol=1e-3, atol=1e-6):
                        matching_keys.append((gt_key, em_key, gt_scale, gt_translation, em_translation))

            # If no match found, that is pretty weird.
            if not matching_keys:
                log.error(f"No EM resolution level matches any GT scale for crop ID '{crop.id}'.")
                continue

            # We get the desired resolution level for the EM volume, labels, and the scale of choice.
            matching_keys.sort(key=lambda x: np.prod(x[2]))
            gt_level, em_level, scale, gt_translation, em_translation = matching_keys[0]

            # Get the desired values for the particular resolution level.
            em_array = em_source_arrays[em_level]
            gt_crop_shape = gt_source_group[f"all/{gt_level}"].shape  # since "all" exists "al"ways, we rely on it.

            log.info(f"Found a resolution match for EM data at level '{em_level}' and GT data at level '{gt_level}'.")

            # Compute the input reference crop from the ground truth metadata.
            starts = gt_translation
            stops = [start + size * vs for start, size, vs in zip(starts, gt_crop_shape, scale)]

            # Get the slices.
            em_starts = [int(round((p_start - em_translation[i]) / scale[i])) for i, p_start in enumerate(starts)]
            em_stops = [int(round((p_stop - em_translation[i]) / scale[i])) for i, p_stop in enumerate(stops)]
            slices = tuple(slice(start, stop) for start, stop in zip(em_starts, em_stops))

        # Pad the slices (in voxel space)
        slices_padded = tuple(
            slice(max(0, sl.start - padding), min(sl.stop + padding, dim), sl.step)
            for sl, dim in zip(slices, em_array.shape)
        )

        # Extract cropped EM volume from remote zarr files.
        em_crop = em_array[tuple(slices_padded)].data.compute()

        # Write all stuff in a crop-level h5 file.
        write_lock = Lock()
        with h5py.File(crop_path, "w") as f:
            # Store metadata
            f.attrs["crop_id"] = crop.id
            f.attrs["scale"] = scale
            f.attrs["em_level"] = em_level

            if gt_level is not None:
                f.attrs["translation"] = gt_translation
                f.attrs["gt_level"] = gt_level

            # Store inputs.
            f.create_dataset(name="raw_crop", data=em_crop, dtype=em_crop.dtype, compression="gzip")
            log.info(f"Saved EM data crop for crop '{crop.id}'.")

            def _fetch_and_write_label(label_name):
                gt_crop = gt_source_group[f"{label_name}/{gt_level}"][:]

                # Next, pad the labels to match the input shape.
                def _pad_to_shape(array):
                    return np.pad(
                        array=array.astype(np.int16),
                        pad_width=[
                            (orig.start - padded.start, padded.stop - orig.stop)
                            for orig, padded in zip(slices, slices_padded)
                        ],
                        mode="constant",
                        constant_values=-1,
                    )

                gt_crop = _pad_to_shape(gt_crop)

                # Write each label to their corresponding hierarchy names.
                with write_lock:
                    f.create_dataset(
                        name=f"label_crop/{label_name}", data=gt_crop, dtype=gt_crop.dtype, compression="gzip"
                    )
                return label_name

            if gt_level is not None:
                # For this one (large) crop in particular, we store labels in serial
                # as multiple threads cannot handle it and silently crash.
                if crop.id == 247:
                    for name in crop_group_inventory:
                        _fetch_and_write_label(name)
                        log.info(f"Saved ground truth crop '{crop.id}' for '{name}'.")
                else:
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
    crops: Union[str, Sequence[str]] = "all",
    resolution: str = "s0",
    padding: int = 64,
    download: bool = False,
) -> Tuple[str, List[str]]:
    """Downloads the CellMap training data.

    Args:
        path: Filepath to a folder where the data will be downloaded for further processing.
        organelles: The choice of organelles to download. By default, loads all types of labels available.
            For one for multiple organelles, specify either like 'mito' or ['mito', 'cell'].
        crops: The choice of crops to download. By default, downloads `all` crops.
            For multiple crops, provide the crop ids as a sequence of crop ids.
        resolution: The choice of resolution in the original volumes.
            By default, downloads the highest resolution: `s0`.
        padding: The choice of padding along each dimensions.
            By default, it pads '64' pixels along all dimensions.
            You can set it to '0' for no padding at all.
            For pixel regions without annotations, it labels the masks with id '-1'.
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
        padding=padding,
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

    assert len(final_crops) > 0, "There seems to be no valid crops in the list."

    return data_path, final_crops


def get_cellmap_paths(
    path: Union[os.PathLike, str],
    organelles: Optional[Union[str, List[str]]] = None,
    crops: Union[str, Sequence[str]] = "all",
    resolution: str = "s0",
    voxel_size: Optional[Tuple[float]] = None,
    padding: int = 64,
    download: bool = False,
    return_test_crops: bool = False,
) -> List[str]:
    """Get the paths to CellMap training data.

    Args:
        path: Filepath to a folder where the data will be downloaded for further processing
        organelles: The choice of organelles to download. By default, loads all types of labels available.
            For one for multiple organelles, specify either like 'mito' or ['mito', 'cell'].
        crops: The choice of crops to download. By default, downloads `all` crops.
            For multiple crops, provide the crop ids as a sequence of crop ids.
        resolution: The choice of resolution in the original volumes.
            By default, downloads the highest resolution: `s0`.
        voxel_size: The choice of voxel size for the preprocessed crops to prepare the dataset.
            By default, chooses all crops in scope.
        padding: The choice of padding along each dimensions.
            By default, it pads '64' pixels along all dimensions.
            You can set it to '0' for no padding at all.
            For pixel regions without annotations, it labels the masks with id '-1'.
        download: Whether to download the data if it is not present.
        return_test_crops: Whether to forcefully return the filepaths of the test crops for other analysis.

    Returns:
        List of the cropped volume data paths.
    """

    if not return_test_crops and ("test" in crops if isinstance(crops, (List, Tuple)) else crops == "test"):
        raise NotImplementedError("The 'test' crops cannot be used in the dataloader.")

    # Get the CellMap data crops.
    data_path, crops = get_cellmap_data(
        path=path, organelles=organelles, crops=crops, resolution=resolution, padding=padding, download=download
    )

    # Get all crops.
    volume_paths = [os.path.join(data_path, f"crop_{c}.h5") for c in crops]

    # Check for valid organelles list to filter crops.
    if organelles is None:
        organelles = "all"

    if isinstance(organelles, str):
        organelles = [organelles]

    other_volume_paths = []
    for organelle in organelles:
        for vpath in volume_paths:
            if f"label_crop/{organelle}" in open_file(vpath) and vpath not in other_volume_paths:
                other_volume_paths.append(vpath)

    if len(other_volume_paths) == 0:
        raise ValueError(f"'{organelles}' are not valid organelle(s) found in the crops: '{crops}'.")

    # Next, we check for valid voxel size to filter crops.
    if voxel_size is None:  # no filtering required.
        another_volume_paths = other_volume_paths
    else:
        another_volume_paths = []
        for vpath in other_volume_paths:
            if all(np.array(voxel_size) == open_file(vpath).attrs["scale"]) and vpath not in another_volume_paths:
                another_volume_paths.append(vpath)

    if len(another_volume_paths) == 0:
        raise ValueError(f"'{voxel_size}' is not a valid voxel size found in the crops: '{crops}'.")

    # Check whether all volume paths exist.
    for volume_path in another_volume_paths:
        if not os.path.exists(volume_path):
            raise FileNotFoundError(f"The volume '{volume_path}' could not be found.")

    return another_volume_paths


def get_cellmap_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    organelles: Optional[Union[str, List[str]]] = None,
    crops: Union[str, Sequence[str]] = "all",
    resolution: str = "s0",
    voxel_size: Optional[Tuple[float]] = None,
    padding: int = 64,
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
        resolution: The choice of resolution in the original volumes.
            By default, downloads the highest resolution: `s0`.
        voxel_size: The choice of voxel size for the preprocessed crops to prepare the dataset.
            By default, chooses all crops in scope.
        padding: The choice of padding along each dimensions.
            By default, it pads '64' pixels along all dimensions.
            You can set it to '0' for no padding at all.
            For pixel regions without annotations, it labels the masks with id '-1'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_cellmap_paths(
        path=path,
        organelles=organelles,
        crops=crops,
        resolution=resolution,
        voxel_size=voxel_size,
        padding=padding, download=download
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
    patch_shape: Tuple[int, ...],
    organelles: Optional[Union[str, List[str]]] = None,
    crops: Union[str, Sequence[str]] = "all",
    resolution: str = "s0",
    voxel_size: Optional[Tuple[float]] = None,
    padding: int = 64,
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
        resolution: The choice of resolution in the original volumes.
            By default, downloads the highest resolution: `s0`.
        voxel_size: The choice of voxel size for the preprocessed crops to prepare the dataset.
            By default, chooses all crops in scope.
        padding: The choice of padding along each dimensions.
            By default, it pads '64' pixels along all dimensions.
            You can set it to '0' for no padding at all.
            For pixel regions without annotations, it labels the masks with id '-1'.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cellmap_dataset(
        path, patch_shape, organelles, crops, resolution, voxel_size, padding, download, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
