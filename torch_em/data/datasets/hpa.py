import os
import json
import shutil
from concurrent import futures
from functools import partial
from glob import glob

import imageio
import h5py
import numpy as np
from PIL import Image, ImageDraw
from skimage import draw as skimage_draw
from skimage import morphology
from tqdm import tqdm

import torch_em
from . import util


URLS = {
    "segmentation": "https://zenodo.org/record/4665863/files/hpa_dataset_v2.zip"
}
CHECKSUMS = {
    "segmentation": "dcd6072293d88d49c71376d3d99f3f4f102e4ee83efb0187faa89c95ec49faa9"
}


def _download_hpa_data(path, name, download):
    os.makedirs(path, exist_ok=True)
    url = URLS[name]
    checksum = CHECKSUMS[name]
    zip_path = os.path.join(path, "data.zip")
    util.download_source(zip_path, url, download=download, checksum=checksum)
    util.unzip(zip_path, path, remove=True)


def _load_features(features):
    # Loop over list and create simple dictionary & get size of annotations
    annot_dict = {}
    skipped = []

    for feat_idx, feat in enumerate(features):
        if feat["geometry"]["type"] not in ["Polygon", "LineString"]:
            skipped.append(feat["geometry"]["type"])
            continue

        # skip empty roi
        if len(feat["geometry"]["coordinates"][0]) <= 0:
            continue

        key_annot = "annot_" + str(feat_idx)
        annot_dict[key_annot] = {}
        annot_dict[key_annot]["type"] = feat["geometry"]["type"]
        annot_dict[key_annot]["pos"] = np.squeeze(
            np.asarray(feat["geometry"]["coordinates"])
        )
        annot_dict[key_annot]["properties"] = feat["properties"]

    # print("Skipped geometry type(s):", skipped)
    return annot_dict


def _generate_binary_masks(annot_dict, shape, erose_size=5, obj_size_rem=500, save_indiv=False):
    # Get dimensions of image and created masks of same size
    # This we need to save somewhere (e.g. as part of the geojson file?)

    # Filled masks and edge mask for polygons
    mask_fill = np.zeros(shape, dtype=np.uint8)
    mask_edge = np.zeros(shape, dtype=np.uint8)
    mask_labels = np.zeros(shape, dtype=np.uint16)

    rr_all = []
    cc_all = []

    if save_indiv is True:
        mask_edge_indiv = np.zeros(
            (shape[0], shape[1], len(annot_dict)), dtype=np.bool
        )
        mask_fill_indiv = np.zeros(
            (shape[0], shape[1], len(annot_dict)), dtype=np.bool
        )

    # Image used to draw lines - for edge mask for freelines
    im_freeline = Image.new("1", (shape[1], shape[0]), color=0)
    draw = ImageDraw.Draw(im_freeline)

    # Loop over all roi
    i_roi = 0
    for roi_key, roi in annot_dict.items():
        roi_pos = roi["pos"]

        # Check region type

        # freeline - line
        if roi["type"] == "freeline" or roi["type"] == "LineString":

            # Loop over all pairs of points to draw the line

            for ind in range(roi_pos.shape[0] - 1):
                line_pos = (
                    roi_pos[ind, 1],
                    roi_pos[ind, 0],
                    roi_pos[ind + 1, 1],
                    roi_pos[ind + 1, 0],
                )
                draw.line(line_pos, fill=1, width=erose_size)

        # freehand - polygon
        elif (
            roi["type"] == "freehand"
            or roi["type"] == "polygon"
            or roi["type"] == "polyline"
            or roi["type"] == "Polygon"
        ):

            # Draw polygon
            rr, cc = skimage_draw.polygon(
                [shape[0] - r for r in roi_pos[:, 1]], roi_pos[:, 0]
            )

            # Make sure it's not outside
            rr[rr < 0] = 0
            rr[rr > shape[0] - 1] = shape[0] - 1

            cc[cc < 0] = 0
            cc[cc > shape[1] - 1] = shape[1] - 1

            # Test if this region has already been added
            if any(np.array_equal(rr, rr_test) for rr_test in rr_all) and any(
                np.array_equal(cc, cc_test) for cc_test in cc_all
            ):
                # print('Region #{} has already been used'.format(i +
                # 1))
                continue

            rr_all.append(rr)
            cc_all.append(cc)

            # Generate mask
            mask_fill_roi = np.zeros(shape, dtype=np.uint8)
            mask_fill_roi[rr, cc] = 1

            # Erode to get cell edge - both arrays are boolean to be used as
            # index arrays later
            mask_fill_roi_erode = morphology.binary_erosion(
                mask_fill_roi, np.ones((erose_size, erose_size))
            )
            mask_edge_roi = (
                mask_fill_roi.astype("int") - mask_fill_roi_erode.astype("int")
            ).astype("bool")

            # Save array for mask and edge
            mask_fill[mask_fill_roi > 0] = 1
            mask_edge[mask_edge_roi] = 1
            mask_labels[mask_fill_roi > 0] = i_roi + 1

            if save_indiv is True:
                mask_edge_indiv[:, :, i_roi] = mask_edge_roi.astype("bool")
                mask_fill_indiv[:, :, i_roi] = mask_fill_roi_erode.astype("bool")

            i_roi = i_roi + 1

        else:
            roi_type = roi["type"]
            raise NotImplementedError(
                f'Mask for roi type "{roi_type}" can not be created'
            )

    del draw

    # Convert mask from free-lines to numpy array
    mask_edge_freeline = np.asarray(im_freeline)
    mask_edge_freeline = mask_edge_freeline.astype("bool")

    # Post-processing of fill and edge mask - if defined
    mask_dict = {}
    if np.any(mask_fill):

        # (1) remove edges , (2) remove small  objects
        mask_fill = mask_fill & ~mask_edge
        mask_fill = morphology.remove_small_objects(
            mask_fill.astype("bool"), obj_size_rem
        )

        # For edge - consider also freeline edge mask

        mask_edge = mask_edge.astype("bool")
        mask_edge = np.logical_or(mask_edge, mask_edge_freeline)

        # Assign to dictionary for return
        mask_dict["edge"] = mask_edge
        mask_dict["fill"] = mask_fill.astype("bool")
        mask_dict["labels"] = mask_labels.astype("uint16")

        if save_indiv is True:
            mask_dict["edge_indiv"] = mask_edge_indiv
            mask_dict["fill_indiv"] = mask_fill_indiv
        else:
            mask_dict["edge_indiv"] = np.zeros(shape + (1,), dtype=np.uint8)
            mask_dict["fill_indiv"] = np.zeros(shape + (1,), dtype=np.uint8)

    # Only edge mask present
    elif np.any(mask_edge_freeline):
        mask_dict["edge"] = mask_edge_freeline
        mask_dict["fill"] = mask_fill.astype("bool")
        mask_dict["labels"] = mask_labels.astype("uint16")

        mask_dict["edge_indiv"] = np.zeros(shape + (1,), dtype=np.uint8)
        mask_dict["fill_indiv"] = np.zeros(shape + (1,), dtype=np.uint8)

    else:
        raise Exception("No mask has been created.")

    return mask_dict


# adapted from
# https://github.com/imjoy-team/kaibu-utils/blob/main/kaibu_utils/__init__.py#L267
def _get_labels(annotation_file, shape, label="*"):
    with open(annotation_file) as f:
        features = json.load(f)["features"]
    if len(features) == 0:
        return np.zeros(shape, dtype="uint16")

    annot_dict_all = _load_features(features)
    annot_types = set(
        annot_dict_all[k]["properties"].get("label", "default")
        for k in annot_dict_all.keys()
    )
    for annot_type in annot_types:
        if label and label != "*" and annot_type != label:
            continue
        # print("annot_type: ", annot_type)
        # Filter the annotations by label
        annot_dict = {
            k: annot_dict_all[k]
            for k in annot_dict_all.keys()
            if label == "*"
            or annot_dict_all[k]["properties"].get("label", "default") == annot_type
        }
        mask_dict = _generate_binary_masks(
            annot_dict, shape,
            erose_size=5,
            obj_size_rem=500,
            save_indiv=True,
        )
        mask = mask_dict["labels"]
        return mask
    raise RuntimeError


def _process_image(in_folder, out_path, channels, with_labels):
    # TODO double check the default order and color matching
    # correspondence to the HPA kaggle data:
    # microtubules: red
    # nuclei: blue
    # er: yellow
    # protein: green
    # default order: rgby = micro, prot, nuclei, er
    all_channels = {"microtubules", "protein", "nuclei", "er"}
    assert len(list(set(channels) - all_channels)) == 0
    raw = []
    for chan in channels:
        im_path = os.path.join(in_folder, f"{chan}.png")
        assert os.path.exists(im_path), im_path
        raw.append(imageio.imread(im_path)[None])
    raw = np.concatenate(raw, axis=0)

    if with_labels:
        annotation_file = os.path.join(in_folder, "annotation.json")
        assert os.path.exists(annotation_file), annotation_file
        labels = _get_labels(annotation_file, raw.shape[1:])
        assert labels.shape == raw.shape[1:]

    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        if with_labels:
            f.create_dataset("labels", data=labels, compression="gzip")


def _process_split(root_in, root_out, channels, n_workers, with_labels):
    os.makedirs(root_out, exist_ok=True)
    inputs = glob(os.path.join(root_in, "*"))
    outputs = [os.path.join(root_out, f"{os.path.split(inp)[1]}.h5") for inp in inputs]
    process = partial(_process_image, channels=channels, with_labels=with_labels)
    with futures.ProcessPoolExecutor(n_workers) as pp:
        list(tqdm(pp.map(process, inputs, outputs), total=len(inputs), desc=f"Process data in {root_in}"))


# save data as h5 with 4 channel raw data and labels extracted from the geo json
def _process_hpa_data(path, channels, n_workers, remove):
    in_path = os.path.join(path, "hpa_dataset_v2")
    assert os.path.exists(in_path), in_path
    for split in ("train", "test", "valid"):
        out_split = "val" if split == "valid" else split
        _process_split(os.path.join(in_path, split), os.path.join(path, out_split),
                       channels=channels, n_workers=n_workers, with_labels=split != "test")
    if remove:
        shutil.rmtree(in_path)


def _check_data(path):
    have_train = len(glob(os.path.join(path, "train", "*.h5"))) == 257
    have_test = len(glob(os.path.join(path, "test", "*.h5"))) == 10
    have_val = len(glob(os.path.join(path, "val", "*.h5"))) == 9
    return have_train and have_test and have_val


def get_hpa_segmentation_dataset(
    path, split, patch_shape,
    offsets=None, boundaries=False, binary=False,
    channels=["microtubules", "protein", "nuclei", "er"],
    download=False, n_workers_preproc=8, **kwargs
):
    data_is_complete = _check_data(path)
    if not data_is_complete:
        _download_hpa_data(path, "segmentation", download)
        _process_hpa_data(path, channels, n_workers_preproc, remove=True)

    kwargs = util.ensure_transforms(ndim=2, **kwargs)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )
    kwargs = util.update_kwargs(kwargs, "ndim", 2)
    kwargs = util.update_kwargs(kwargs, "with_channels", True)

    paths = glob(os.path.join(path, split, "*.h5"))
    raw_key = "raw"
    label_key = "labels"

    return torch_em.default_segmentation_dataset(paths, raw_key, paths, label_key, patch_shape, **kwargs)


def get_hpa_segmentation_loader(
    path, split, patch_shape, batch_size,
    offsets=None, boundaries=False, binary=False,
    channels=["microtubules", "protein", "nuclei", "er"],
    download=False, n_workers_preproc=8, **kwargs
):
    ds_kwargs, loader_kwargs = util.split_kwargs(
        torch_em.default_segmentation_dataset, **kwargs
    )
    dataset = get_hpa_segmentation_dataset(
        path, split, patch_shape,
        offsets=offsets, boundaries=boundaries, binary=binary,
        channels=channels, download=download, n_workers_preproc=n_workers_preproc,
        **ds_kwargs
    )
    loader = torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
