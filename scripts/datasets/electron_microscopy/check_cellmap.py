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


def _download_cellmap_data(path, crops="all", resolution="s0"):
    """Downloads the CellMap data.

    Args:
        crops: A string of all crops to download or choose 'all' or 'test'.
            For crops, the crops are expected such that '111,112,113', etc.
    """
    # Import packages.
    import time
    import structlog

    import h5py

    from xarray import DataArray
    from xarray_ome_ngff import read_multiscale_group

    from cellmap_segmentation_challenge.utils.fetch_data import read_group, subset_to_slice
    from cellmap_segmentation_challenge.utils.crops import fetch_crop_manifest, get_test_crops

    # Some important stuff.
    padding = 0
    fetch_save_start = time.time()
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

    for crop in crops_parsed:
        log = log.bind(crop_id=crop.id, dataset=crop.dataset)

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

        gt_crop_shape = gt_source_group[f"all/{resolution}"].shape  # since "all" exists always, we rely on this.

        # Get the EM volume of highest resolution.
        em_source_group = read_group(str(crop.em_url), storage_options={"anon": True})
        log.info(f"Found EM data at {crop.em_url}.")

        # Get the multiscale model of the source em group
        array_wrapper = {"name": "dask_array", "config": {"chunks": "auto"}}
        em_source_arrays = read_multiscale_group(em_source_group, array_wrapper)
        em_s0 = em_source_arrays[resolution]

        scale, translation = _get_scale_and_translation(gt_source_group["all"])
        if scale is None and translation is None:
            raise RuntimeError

        # Compute the input reference crop from the ground truth metadata.
        starts = translation
        stops = [start + size * vs for start, size, vs in zip(translation, gt_crop_shape, scale)]

        # Get the slices.
        coords = {dim: np.array([start, stop]) for dim, (start, stop) in zip(em_s0.dims, zip(starts, stops))}
        slices = subset_to_slice(em_s0, DataArray(dims=em_s0.dims, coords=coords))

        # Pad the slices (in voxel space)
        slices_padded = tuple(
            slice(max(0, sl.start - padding), min(sl.stop + padding, dim), sl.step)
            for sl, dim in zip(slices, em_s0.shape)
        )

        # Extract cropped EM volume from remote zarr files.
        em_crop = em_s0[tuple(slices_padded)].data.compute()

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock
        write_lock = Lock()

        # Write all stuff in a crop-level h5 file.
        crop_path = dest_path_abs / f"crop_{crop.id}.h5"
        with h5py.File(crop_path, "w") as f:
            # Store metadata
            f.attrs["crop_id"] = crop.id
            f.attrs["scale"] = scale
            f.attrs["translation"] = translation

            # Store inputs.
            f.create_dataset(name="raw_crop", shape=em_crop.shape, data=em_crop, compression="gzip")

            def fetch_and_write_label(label_name):
                gt_crop = gt_source_group[f"{label_name}/{resolution}"][:]
                with write_lock:
                    f.create_dataset(
                        name=f"label_crop/{label_name}",
                        shape=gt_crop.shape,
                        data=gt_crop,
                        compression="gzip",
                    )
                return label_name

            with ThreadPoolExecutor() as pool:
                futures = {pool.submit(fetch_and_write_label, name): name for name in crop_group_inventory}
                for future in as_completed(futures):
                    label_name = future.result()
                    log.info(f"Saved ground truth crop '{crop.id}' for '{label_name}'.")

        log.info(f"Saved crop {crop.id} to {crop_path}")
        log = log.unbind("crop_id", "dataset")

    log.info(f"Done after {time.time() - fetch_save_start:0.3f}s")
    log.info(f"Data saved to {dest_path_abs}")


def main():
    # ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/cellmap-challenge"
    ROOT = "/media/anwai/ANWAI/data/cellmap-challenge"

    # _test_loading(os.path.join(ROOT, "data_matched_res_no_pad"))
    _download_cellmap_data(path=os.path.join(ROOT, "data"), crops="234", resolution="s0")


if __name__ == "__main__":
    main()
