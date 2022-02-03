import os
from .util import download_source, unzip


URLS = {
    "segmentation": "https://zenodo.org/record/4665863/files/hpa_dataset_v2.zip"
}
CHECKSUMS = {
    "segmentation": None
}


def _require_hpa_data(path, name, download):
    os.makedirs(path, exist_ok=True)
    url = URLS[name]
    checksum = CHECKSUMS[name]
    zip_path = os.path.join(path, "data.zip")
    download_source(zip_path, url, download=download, checksum=checksum)
    # unzip(zip_path, path, remove=True)


def get_hpa_segmentation_loader(path, patch_shape, split,
                                download=False, **kwargs):
    data_is_complete = False
    if not data_is_complete:
        _require_hpa_data(path, "segmentation", download)
