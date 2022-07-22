import os
from shutil import rmtree

from .util import download_source, unzip, update_kwargs

# TODO find a source for this!
URL = ""
CHECKSUM = ""


def _require_lucchi_data(path, download):
    expected_paths = [
        os.path.join(path, "epfl_train.h5"),
        os.path.join(path, "epfl_val.h5"),
        os.path.join(path, "epfl_test.h5"),
    ]
    # download and unzip the data
    if os.path.exists(path):
        assert all(os.path.exists(pp) for pp in expected_paths)
        return path

    os.makedirs(path)
    tmp_path = os.path.join(path, "epfl.zip")
    download_source(tmp_path, URL, download, checksum=CHECKSUM)
    unzip(tmp_path, path, remove=True)
    rmtree(tmp_path)

    assert all(os.path.exists(pp) for pp in expected_paths)


def get_lucchi_loader():
    pass
