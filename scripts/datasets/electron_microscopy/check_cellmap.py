import os
from glob import glob

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

            label_crop = f[f"recon-1/labels/groundtruth/{crop_name}/{label_choice}/s0"][:]
            image_crop = _get_matching_crop(f["recon-1/em/fibsem-uint8/s0"], offset, label_crop.shape)

            # Visualize image and corresponding label crops.
            import napari
            v = napari.Viewer()
            v.add_image(image_crop, name="Image")
            v.add_labels(label_crop, name="Labels")
            napari.run()


def main():
    # ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/cellmap-challenge"
    ROOT = "/media/anwai/ANWAI/data/cellmap-challenge"

    data_root = os.path.join(ROOT, "data_matched_res_no_pad")

    _test_loading(data_root)


if __name__ == "__main__":
    main()
