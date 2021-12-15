import os
import hashlib
import zipfile
from shutil import copyfileobj
from warnings import warn

import requests
from tqdm import tqdm

BIOIMAGEIO_IDS = {
    "covid_if": "ilastik/covid_if_training_data",
    "cremi": "ilastik/cremi_training_data",
    "dsb": "ilastik/stardist_dsb_training_data",
    "isbi2012": "ilastik/isbi2012_neuron_segmentation_challenge",
    "livcell": "ilastik/livecell_dataset",
    "mitoem": "ilastik/mitoem_segmentation_challenge",
    "monuseg": "deepimagej/monuseg_digital_pathology_miccai2018",
    "ovules": "",  # not on bioimageio yet
    "platynereis": "ilastik/platynereis_em_training_data"
}


def get_bioimageio_dataset_id(dataset_name):
    assert dataset_name in BIOIMAGEIO_IDS
    return BIOIMAGEIO_IDS[dataset_name]


def get_checksum(filename):
    with open(filename, "rb") as f:
        file_ = f.read()
        checksum = hashlib.sha256(file_).hexdigest()
    return checksum


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

    if checksum is not None:
        this_checksum = get_checksum(path)
        if this_checksum != checksum:
            raise RuntimeError(
                "The checksum of the download does not match the expected checksum."
                f"Expected: {checksum}, got: {this_checksum}"
            )
        print("Download successfull and checksums agree.")
    else:
        warn("The file was downloaded, but no checksum was provided, so the file may be corrupted.")


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
