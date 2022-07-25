import os
import re
from concurrent import futures
from glob import glob
from shutil import rmtree

import imageio
import h5py
import numpy as np
import torch_em
from tqdm import tqdm
from .util import download_source, update_kwargs, unzip

URL = "https://www.repository.cam.ac.uk/bitstream/handle/1810/262530/PNAS.zip?sequence=4&isAllowed=y"
CHECKSUM = "39341398389baf6d93c3f652b7e2e8aedc5579c29dfaf2b82b41ebfc3caa05c4"


def _sort_time(paths):
    tps = [int(os.path.basename(path).split("_")[0][:-3]) for path in paths]
    time_sorted = np.argsort(tps)
    return {tps[i]: paths[i] for i in time_sorted}


def _sort_time_nucseg(paths):

    def get_tp(fname):
        pattern = "t[0-9]+"
        tpstring = re.findall(pattern, fname)
        assert len(tpstring) == 1
        return int(tpstring[0][1:])

    tps = [get_tp(os.path.basename(path)) for path in paths]
    time_sorted = np.argsort(tps)
    return {tps[i]: paths[i] for i in time_sorted}


def _process_plant(input_folder, out_folder, with_nuclei, n_threads):
    os.makedirs(out_folder, exist_ok=True)

    membrane_raw = glob(os.path.join(input_folder, "processed_tiffs", "*trim-acylYFP*"))
    assert len(membrane_raw) > 0
    membrane_raw = _sort_time(membrane_raw)

    membrane_labels = glob(os.path.join(input_folder, "segmentation_tiffs", "*trim-acylYFP*"))
    assert len(membrane_labels) > 0
    membrane_labels = _sort_time(membrane_labels)

    if with_nuclei:
        nucleus_raw = glob(os.path.join(input_folder, "processed_tiffs", "*trim-clv3*"))
        assert len(nucleus_raw) > 0
        nucleus_raw = _sort_time(nucleus_raw)

        nucleus_labels = glob(os.path.join(input_folder, "nuclear_segmentation_tiffs", "*.tif"))
        assert len(nucleus_labels) > 0
        nucleus_labels = _sort_time_nucseg(nucleus_labels)

    def process_tp(tp):
        # check if we have any labeled data
        if tp not in membrane_labels:
            if not with_nuclei:
                return
            if with_nuclei and tp not in nucleus_labels:
                return

        out_path = os.path.join(out_folder, f"tp{tp:03}.h5")
        with h5py.File(out_path, "w") as f:
            raw = imageio.volread(membrane_raw[tp])
            labels = imageio.volread(membrane_labels[tp])
            labels -= 1
            assert labels.min() == 0
            assert raw.shape == labels.shape
            f.create_dataset("raw/membrane", data=raw, compression="gzip")
            f.create_dataset("labels/membrane", data=labels, compression="gzip")
            if with_nuclei and tp in nucleus_labels:
                raw = imageio.volread(nucleus_raw[tp])
                labels = imageio.volread(nucleus_labels[tp])
                labels -= 1
                assert labels.min() == 0
                assert raw.shape == labels.shape
                f.create_dataset("raw/nucleus", data=raw, compression="gzip")
                f.create_dataset("labels/nucleus", data=labels, compression="gzip")

    tps = list(membrane_raw.keys())
    tps.sort()
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(process_tp, tps), total=len(tps), desc=f"Process plant: {out_folder}"
        ))


def _process_pnas_data(path):
    root = os.path.join(path, "PNAS")
    plants = glob(os.path.join(root, "plant*"))
    plants.sort()
    # we sort into plants with and without nucleus data
    # and reorder them accrodingly
    plants_nuclei = [plant for plant in plants
                     if os.path.exists(os.path.join(plant, "nuclear_segmentation_tiffs"))]
    plants_no_nuclei = list(set(plants) - set(plants_nuclei))

    n_threads = 8
    for i, plant in enumerate(plants_nuclei):
        out_folder = os.path.join(path, f"plant{i}")
        _process_plant(plant, out_folder, with_nuclei=True, n_threads=n_threads)

    for i, plant in enumerate(plants_no_nuclei, start=len(plants_nuclei)):
        out_folder = os.path.join(path, f"plant{i}")
        _process_plant(plant, out_folder, with_nuclei=False, n_threads=n_threads)

    rmtree(root)


def _require_pnas_data(path, download):
    # download and unzip the data
    if os.path.exists(path):
        return path

    os.makedirs(path)
    tmp_path = os.path.join(path, "pnas.zip")
    download_source(tmp_path, URL, download, checksum=CHECKSUM)
    unzip(tmp_path, path, remove=True)

    _process_pnas_data(path)


def get_plants_with_membrane_labels(path):
    plant_dirs = glob(os.path.join(path, "plant*"))
    # all plants have membrane labels
    plant_ids = [int(os.path.basename(pdir)[5:]) for pdir in plant_dirs]
    plant_ids.sort()
    return plant_ids


def _get_membrane_paths(path, plant_ids):
    all_plant_ids = get_plants_with_membrane_labels(path)
    if plant_ids is None:
        plant_ids = all_plant_ids
    else:
        invalid_plant_ids = list(set(plant_ids) - set(all_plant_ids))
        if len(invalid_plant_ids) > 0:
            raise ValueError(f"Invalid plant ids for membrane loader: {invalid_plant_ids}")
    paths = []
    for plant_id in plant_ids:
        this_paths = glob(os.path.join(os.path.join(path, f"plant{plant_id}", "*.h5")))
        # we don't have guarantees that all the files have membrane labels, so we check for each individual path
        for ppath in this_paths:
            with h5py.File(ppath, "r") as f:
                if "labels/membrane" in f:
                    paths.append(ppath)
    return paths


def get_pnas_membrane_loader(
    path,
    plant_ids=None,
    download=False,
    offsets=None,
    boundaries=False,
    binary=False,
    ndim=3,
    **kwargs
):
    _require_pnas_data(path, download)
    paths = _get_membrane_paths(path, plant_ids)

    assert not (offsets is not None and boundaries)
    if offsets is not None:
        # we add a binary target channel for foreground background segmentation
        label_transform = torch_em.transform.label.AffinityTransform(offsets=offsets,
                                                                     add_binary_target=binary,
                                                                     add_mask=True)
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform2", label_transform, msg=msg)
    elif boundaries:
        label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=binary)
        msg = "Boundaries is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)
    elif binary:
        label_transform = torch_em.transform.label.labels_to_binary
        msg = "Binary is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)

    raw_key, label_key = "raw/membrane", "labels/membrane"
    return torch_em.default_segmentation_loader(
        paths, raw_key,
        paths, label_key,
        ndim=ndim, is_seg_dataset=True,
        **kwargs
    )


def get_plants_with_nucleus_labels(path):

    def check_nuc_labels(plant_dir):
        paths = glob(os.path.join(plant_dir, "*.h5"))
        for path in paths:
            with h5py.File(path, "r") as f:
                if "labels/nucleus" in f:
                    return True
        return False

    plant_dirs = glob(os.path.join(path, "plant*"))
    plant_dirs = [plant_dir for plant_dir in plant_dirs if check_nuc_labels(plant_dir)]
    plant_ids = [int(os.path.basename(pdir)[5:]) for pdir in plant_dirs]
    plant_ids.sort()
    return plant_ids


def _get_nucleus_paths(path, plant_ids):
    all_plant_ids = get_plants_with_nucleus_labels(path)
    if plant_ids is None:
        plant_ids = all_plant_ids
    else:
        invalid_plant_ids = list(set(plant_ids) - set(all_plant_ids))
        if len(invalid_plant_ids) > 0:
            raise ValueError(f"Invalid plant ids for membrane loader: {invalid_plant_ids}")
    paths = []
    for plant_id in plant_ids:
        this_paths = glob(os.path.join(os.path.join(path, f"plant{plant_id}", "*.h5")))
        # we don't have guarantees that all the files have nucleus labels, so we check for each individual path
        for ppath in this_paths:
            with h5py.File(ppath, "r") as f:
                if "labels/nucleus" in f:
                    paths.append(ppath)
    return paths


def get_pnas_nucleus_loader(
    path,
    plant_ids=None,
    download=False,
    offsets=None,
    boundaries=False,
    binary=False,
    ndim=3,
    **kwargs
):
    _require_pnas_data(path, download)
    paths = _get_nucleus_paths(path, plant_ids)

    assert not (offsets is not None and boundaries)
    if offsets is not None:
        # we add a binary target channel for foreground background segmentation
        label_transform = torch_em.transform.label.AffinityTransform(offsets=offsets,
                                                                     add_binary_target=binary,
                                                                     add_mask=True)
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform2", label_transform, msg=msg)
    elif boundaries:
        label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=binary)
        msg = "Boundaries is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)
    elif binary:
        label_transform = torch_em.transform.label.labels_to_binary
        msg = "Binary is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)

    raw_key, label_key = "raw/nucleus", "labels/nucleus"
    return torch_em.default_segmentation_loader(
        paths, raw_key,
        paths, label_key,
        ndim=ndim, is_seg_dataset=True,
        **kwargs
    )
