import os
from glob import glob
from pathlib import Path

import numpy as np


def _get_scale_and_translation(label_zobj, resolution="s0"):
    all_scales = label_zobj.attrs.get("multiscales", [])[0]
    for ds in all_scales.get("datasets", []):
        if ds.get("path") == resolution:
            transforms = ds.get("coordinateTransformations", [])
            scale = next((t["scale"] for t in transforms if t["type"] == "scale"), None)
            translation = next((t["translation"] for t in transforms if t["type"] == "translation"), None)
            return scale, translation
    return None, None


def _get_matching_crop(image_zobj, offset, crop_shape):
    z, y, x = offset
    dz, dy, dx = crop_shape
    return image_zobj[z:z+dz, y:y+dy, x:x+dx]


def _test_loading(data_root):
    # Open each data directory
    for data_dir in glob(os.path.join(data_root, "*")):
        # Get the zarr files inside each data directory.
        data_path = os.path.join(data_dir, f"{os.path.basename(data_dir)}.zarr")

        # Open the zarr file.
        import zarr
        f = zarr.open(data_path, mode="r")

        # Image
        print(list(f["recon-1/em/fibsem-uint8"].keys()))
        print(f["recon-1/em/fibsem-uint8/s0"])

        # Corresponding labels
        label_choice = "all"
        print(list(f["recon-1/labels/groundtruth"].keys()))
        print(f[f"recon-1/labels/groundtruth/crop234/{label_choice}/s0"])

        # Map labels back to the original spatial shapes.
        for crop_name in list(f["recon-1/labels/groundtruth"].keys()):

            # Get the offset values from the translation and scale values given.
            scale, translation = _get_scale_and_translation(
                f[f"recon-1/labels/groundtruth/{crop_name}/{label_choice}"], "s0"
            )

            if scale is None and translation is None:
                raise RuntimeError

            offset = (np.array(translation) / np.array(scale)).astype(int)
            print(offset)

            view = False
            if view:
                label_crop = f[f"recon-1/labels/groundtruth/{crop_name}/{label_choice}/s0"][:]
                image_crop = _get_matching_crop(f["recon-1/em/fibsem-uint8/s0"], offset, label_crop.shape)

                # Visualize image and corresponding label crops.
                import napari
                v = napari.Viewer()
                v.add_image(image_crop, name="Image")
                v.add_labels(label_crop, name="Labels")
                napari.run()


def _download_cellmap_data(path, crops="all", fetch_all_em_resolutions=False):
    """Downloads the CellMap data.

    Args:
        crops: A string of all crops to download or choose 'all' or 'test'.
            For crops, the crops are expected such that '111,112,113', etc.
    """
    # Import packages.
    import time
    import structlog
    from yarl import URL
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import zarr
    from zarr.storage import FSStore
    from pydantic_zarr.v2 import GroupSpec

    from xarray import DataArray
    from xarray_ome_ngff import read_multiscale_group
    from xarray_ome_ngff.v04.multiscale import transforms_from_coords, VectorScale

    from cellmap_segmentation_challenge.utils.crops import (
        fetch_crop_manifest, get_test_crops, TestCropRow,
    )
    from cellmap_segmentation_challenge.utils.fetch_data import (
        read_group,
        _resolve_em_dest_path,
        _resolve_gt_dest_path,
        partition_copy_store,
        get_chunk_keys,
        subset_to_slice,
    )

    # Some important stuff.
    batch_size = 256
    num_workers = 32
    raw_padding = 0
    fetch_save_start = time.time()
    mode = "a"  # The access mode for getting data.
    pool = ThreadPoolExecutor(max_workers=num_workers)
    log = structlog.get_logger()

    dest_path_abs = Path(path).absolute()
    dest_path_abs.mkdir(exist_ok=True)

    # Get the entire crop manifest
    crops_from_manifest = fetch_crop_manifest()

    # Get the desired crop info from the manifest.
    if crops in ["all", "test"]:
        test_crops = get_test_crops()
        log.info(f"Found '{len(test_crops)}' test crops.")

    if crops == "all":
        crops_parsed = crops_from_manifest + test_crops
    elif crops == "test":
        crops_parsed = test_crops
    else:
        crops_split = tuple(int(x) for x in crops.split(","))
        crops_parsed = tuple(filter(lambda v: v.id in crops_split, crops_from_manifest))

    # Now get the crop ids.
    if len(crops_parsed) == 0:
        log.info(f"No crops found matching '{crops}'. Doing nothing.")
        return

    crop_ids = tuple(c.id for c in crops_parsed)
    log.info(f"Preparing to copy the following crops: '{crop_ids}'.")
    log.info(f"Data will be saved to '{dest_path_abs}'.")

    futures = []
    for crop in crops_parsed:
        log = log.bind(crop_id=crop.id, dataset=crop.dataset)

        # Create a destination path for the zarr files.
        dest_root = URL.build(
            scheme="file", path=f"/{dest_path_abs.as_posix().lstrip('/')}"
        ).joinpath(f"{crop.dataset}/{crop.dataset}.zarr")

        # Get the cropped labels.
        if isinstance(crop.gt_source, TestCropRow):
            log.info(f"Test crop {crop.id} does not have GT data. Fetching em data only.")
        else:
            # Get the source URL for cropped label volume.
            gt_source_url = crop.gt_source
            log.info(f"Fetching GT data for crop '{crop.id}' from '{gt_source_url}'.")

            try:
                gt_source_group = read_group(str(gt_source_url), storage_options={"anon": True})
                log.info(f"Found GT data at '{gt_source_url}'.")

                gt_dest_path = _resolve_gt_dest_path(crop)
                dest_root_group = zarr.open_group(str(dest_root), mode=mode)

                # Create intermediate groups
                dest_root_group.require_group(gt_dest_path)
                dest_crop_group = zarr.open_group(str(dest_root / gt_dest_path).replace("%5C", "\\"), mode=mode)

                fs = gt_source_group.store.fs
                store_path = gt_source_group.store.path

                # NOTE: Using fs.find here is a performance hack until we fix the slowness of traversing the
                # zarr hierarchy to build the list of files.
                gt_files = fs.find(store_path)
                crop_group_inventory = tuple(fn.split(store_path)[-1] for fn in gt_files)
                log.info(f"Preparing to fetch '{len(crop_group_inventory)}' files from '{gt_source_url}'.")

                futures.extend(
                    partition_copy_store(
                        keys=crop_group_inventory,
                        source_store=gt_source_group.store,
                        dest_store=dest_crop_group.store,
                        batch_size=batch_size,
                        pool=pool,
                    )
                )
            except zarr.errors.GroupNotFoundError:
                log.info(f"No Zarr group was found at '{gt_source_url}'. This crop will be skipped.")
                continue

        # Get the source URL for the input volume.
        em_source_url = crop.em_url
        padding = raw_padding

        try:
            em_source_group = read_group(str(em_source_url), storage_options={"anon": True})
            log.info(f"Found EM data at '{em_source_url}'.")

            # Model the em group locally.
            em_dest_path = _resolve_em_dest_path(crop)
            dest_em_group = GroupSpec.from_zarr(em_source_group).to_zarr(
                FSStore(str(dest_root / em_dest_path).replace("%5C", "\\")), path="", overwrite=(mode == "w"),
            )

            # Get the multi-scale model of the source em group.
            array_wrapper = {"name": "dask_array", "config": {"chunks": "auto"}}
            em_source_arrays = read_multiscale_group(em_source_group, array_wrapper=array_wrapper)

            # Get the overlapping region between the crop and the full array, in array coordinates.
            em_group_inventory = ()
            em_source_arrays_sorted = sorted(
                em_source_arrays.items(), key=lambda kv: np.prod(kv[1].shape), reverse=True
            )

            crop_multiscale_group = None
            if isinstance(crop.gt_source, TestCropRow):
                base_gt_scale = VectorScale(scale=crop.gt_source.voxel_size)
            else:
                for _, group in gt_source_group.groups():
                    try:
                        crop_multiscale_group = read_multiscale_group(group, array_wrapper=array_wrapper)
                        break
                    except (ValueError, TypeError):
                        continue

                if crop_multiscale_group is None:
                    log.info(f"No multiscale groups found in '{gt_source_url}'. No EM data can be fetched.")
                    continue

                gt_source_arrays_sorted = sorted(
                    crop_multiscale_group.items(), key=lambda kv: np.prod(kv[1].shape), reverse=True,
                )
                _, (base_gt_scale, _) = transforms_from_coords(
                    gt_source_arrays_sorted[0][1].coords, transform_precision=4
                )

            # Decide whether we get all resolutions or the highest resolution.
            if fetch_all_em_resolutions:
                ratio_threshold = 0
            else:
                ratio_threshold = 0.8

            # Let's download the volumes now.
            none_yet = True
            for key, array in em_source_arrays_sorted:
                em_group_inventory += (f"{key}/.zarray",)
                if any(len(coord) <= 1 for coord in array.coords.values()):
                    log.info(f"Skipping scale level '{key}' because it has no spatial dimensions")
                    continue

                _, (current_scale, _) = transforms_from_coords(array.coords, transform_precision=4)
                scale_ratios = tuple(
                    s_gt / s_current for s_current, s_gt in zip(current_scale.scale, base_gt_scale.scale)
                )

                if all(tuple(x > ratio_threshold for x in scale_ratios)):
                    # # Relative padding based on the scale of the current resolution:
                    # relative_scale = base_em_scale.scale[0] / current_scale.scale[0]
                    # current_pad = int(padding * relative_scale) # Padding relative to the current scale

                    # Uniform voxel padding for all scales:
                    current_pad = padding
                    if isinstance(crop.gt_source, TestCropRow):
                        starts = crop.gt_source.translation
                        stops = tuple(
                            start + size * vs
                            for start, size, vs in zip(starts, crop.gt_source.shape, crop.gt_source.voxel_size)
                        )
                        coords = array.coords.copy()
                        for k, v in zip(array.coords.keys(), np.array((starts, stops)).T):
                            coords[k] = v

                        slices = subset_to_slice(
                            array,
                            DataArray(dims=array.dims, coords=coords),
                            # force_nonempty=none_yet,
                        )
                    else:
                        slices = subset_to_slice(
                            array,
                            crop_multiscale_group["s0"],
                            # force_nonempty=none_yet,
                        )

                    slices_padded = tuple(
                        slice(max(sl.start - current_pad, 0), min(sl.stop + current_pad, shape), sl.step)
                        for sl, shape in zip(slices, array.shape)
                    )

                    new_chunks = tuple(map(lambda v: f"{key}/{v}", get_chunk_keys(em_source_group[key], slices_padded)))
                    log.debug(f"Gathering {len(new_chunks)} chunks from level {key}.")
                    none_yet = none_yet and len(new_chunks) == 0
                    em_group_inventory += new_chunks
                else:
                    log.info(
                        f"Skipping scale level '{key}' because it is sampled more densely than the groundtruth data"
                    )

            # em_group_inventory += (".zattrs",)
            # assert not none_yet, "No EM data was found for any resolution level."
            log.info(f"Preparing to fetch '{len(em_group_inventory)}' files from '{em_source_url}'.")
            futures.extend(
                partition_copy_store(
                    keys=em_group_inventory,
                    source_store=em_source_group.store,
                    dest_store=dest_em_group.store,
                    batch_size=batch_size,
                    pool=pool,
                )
            )

        except zarr.errors.GroupNotFoundError:
            log.info(f"No EM data was found at '{em_source_url}'. Saving EM data will be skipped.")
            continue

    log = log.unbind("crop_id", "dataset")
    log = log.bind(save_location=dest_path_abs)
    num_iter = len(futures)
    for idx, maybe_result in enumerate(as_completed(futures)):
        try:
            _ = maybe_result.result()
            log.debug(f"Completed fetching batch {idx + 1} / {num_iter}")
        except Exception as e:
            log.exception(e)

    log.unbind("save_location")
    log.info(f"Done after {time.time() - fetch_save_start:0.3f}s")
    log.info(f"Data saved to {dest_path_abs}")


def main():
    # ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/cellmap-challenge"
    ROOT = "/media/anwai/ANWAI/data/cellmap-challenge"

    # _test_loading(os.path.join(ROOT, "data_matched_res_no_pad"))
    _download_cellmap_data(path=os.path.join(ROOT, "data"), crops="234", fetch_all_em_resolutions=False)


if __name__ == "__main__":
    main()
