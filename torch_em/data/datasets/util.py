import inspect
import os
import hashlib
import zipfile
from shutil import copyfileobj
from warnings import warn

import gdown
import torch
import torch_em
import requests

from tqdm import tqdm

BIOIMAGEIO_IDS = {
    "covid_if": "ilastik/covid_if_training_data",
    "cremi": "ilastik/cremi_training_data",
    "dsb": "ilastik/stardist_dsb_training_data",
    "hpa": "",  # not on bioimageio yet
    "isbi2012": "ilastik/isbi2012_neuron_segmentation_challenge",
    "kasthuri": "",  # not on bioimageio yet:
    "livecell": "ilastik/livecell_dataset",
    "lucchi": "",  # not on bioimageio yet:
    "mitoem": "ilastik/mitoem_segmentation_challenge",
    "monuseg": "deepimagej/monuseg_digital_pathology_miccai2018",
    "ovules": "",  # not on bioimageio yet
    "plantseg_root": "ilastik/plantseg_root",
    "plantseg_ovules": "ilastik/plantseg_ovules",
    "platynereis": "ilastik/platynereis_em_training_data",
    "snemi": "",  # not on bioimagegio yet
    "uro_cell": "",  # not on bioimageio yet: https://doi.org/10.1016/j.compbiomed.2020.103693
    "vnc": "ilastik/vnc",
}


def get_bioimageio_dataset_id(dataset_name):
    assert dataset_name in BIOIMAGEIO_IDS
    return BIOIMAGEIO_IDS[dataset_name]


def get_checksum(filename):
    with open(filename, "rb") as f:
        file_ = f.read()
        checksum = hashlib.sha256(file_).hexdigest()
    return checksum


def _check_checksum(path, checksum):
    if checksum is not None:
        this_checksum = get_checksum(path)
        if this_checksum != checksum:
            raise RuntimeError(
                "The checksum of the download does not match the expected checksum."
                f"Expected: {checksum}, got: {this_checksum}"
            )
        print("Download successful and checksums agree.")
    else:
        warn("The file was downloaded, but no checksum was provided, so the file may be corrupted.")


# this needs to be extended to support download from s3 via boto,
# if we get a resource that is available via s3 without support for http
def download_source(path, url, download, checksum=None, verify=True):
    if os.path.exists(path):
        return
    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False")

    with requests.get(url, stream=True, verify=verify) as r:
        if r.status_code != 200:
            r.raise_for_status()
            raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
        file_size = int(r.headers.get("Content-Length", 0))
        desc = f"Download {url} to {path}"
        if file_size == 0:
            desc += " (unknown file size)"
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw, open(path, "wb") as f:
            copyfileobj(r_raw, f)

    _check_checksum(path, checksum)


def download_source_gdrive(path, url, download, checksum=None):
    if os.path.exists(path):
        return
    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False")

    gdown.download(url, path, quiet=False)
    _check_checksum(path, checksum)


def update_kwargs(kwargs, key, value, msg=None):
    if key in kwargs:
        msg = f"{key} will be over-ridden in loader kwargs." if msg is None else msg
        warn(msg)
    kwargs[key] = value
    return kwargs


def unzip(zip_path, dst, remove=True):
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(dst)
    if remove:
        os.remove(zip_path)


def split_kwargs(function, **kwargs):
    function_parameters = inspect.signature(function).parameters
    parameter_names = list(function_parameters.keys())
    other_kwargs = {k: v for k, v in kwargs.items() if k not in parameter_names}
    kwargs = {k: v for k, v in kwargs.items() if k in parameter_names}
    return kwargs, other_kwargs


# this adds the default transforms for 'raw_transform' and 'transform'
# in case these were not specified in the kwargs
# this is NOT necessary if 'default_segmentation_dataset' is used, only if a dataset class
# is used directly, e.g. in the LiveCell Loader
def ensure_transforms(ndim, **kwargs):
    if "raw_transform" not in kwargs:
        kwargs = update_kwargs(kwargs, "raw_transform", torch_em.transform.get_raw_transform())
    if "transform" not in kwargs:
        kwargs = update_kwargs(kwargs, "transform", torch_em.transform.get_augmentations(ndim=ndim))
    return kwargs


def add_instance_label_transform(
    kwargs, add_binary_target, label_dtype=None, binary=False, boundaries=False, offsets=None
):
    assert sum((offsets is not None, boundaries, binary)) <= 1
    if offsets is not None:
        label_transform2 = torch_em.transform.label.AffinityTransform(offsets=offsets,
                                                                      add_binary_target=add_binary_target,
                                                                      add_mask=True)
        msg = "Offsets are passed, but 'label_transform2' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform2", label_transform2, msg=msg)
        label_dtype = torch.float32
    elif boundaries:
        label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=add_binary_target)
        msg = "Boundaries is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)
        label_dtype = torch.float32
    elif binary:
        label_transform = torch_em.transform.label.labels_to_binary
        msg = "Binary is set to true, but 'label_transform' is in the kwargs. It will be over-ridden."
        kwargs = update_kwargs(kwargs, "label_transform", label_transform, msg=msg)
        label_dtype = torch.float32
    return kwargs, label_dtype
