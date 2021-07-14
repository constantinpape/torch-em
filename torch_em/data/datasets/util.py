import os
import hashlib
import zipfile
from shutil import copyfileobj
from warnings import warn

import requests

BIOIMAGEIO_IDS = {
    "covid_if": "",
    "cremi": "",
    "dsb": "",
    "isbi2012": "",
    "mitoem": "",
    "ovules": "",
    "platynereis": ""
}


def get_bioimageio_dataset_id(dataset_name):
    assert dataset_name in BIOIMAGEIO_IDS
    return BIOIMAGEIO_IDS[dataset_name]


def get_checksum(filename):
    with open(filename, "rb") as f:
        file_ = f.read()
        checksum = hashlib.sha256(file_).hexdigest()
    return checksum


# TODO
# - allow for s3 links and use boto3 or s3fs to download
def download_source(path, url, download, checksum=None):
    if os.path.exists(path):
        return
    if not download:
        raise RuntimeError(f"Cannot find the data at {path}, but download was set to False")

    print("Download file fron", url, "to", path)
    with requests.get(url, stream=True) as r:
        with open(path, 'wb') as f:
            copyfileobj(r.raw, f)

    if checksum is not None:
        this_checksum = get_checksum(path)
        if this_checksum != checksum:
            raise RuntimeError("The checksum of the download does not match the expected checksum.")
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
    with zipfile.ZipFile(zip_path, 'r') as f:
        f.extractall(dst)
    if remove:
        os.remove(zip_path)
