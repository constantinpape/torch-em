import os
import h5py
import zipfile
import numpy as np
from glob import glob

import torch_em


URL = None  # TODO: here - https://datasets.deepcell.org/data


def _unzip_data(zip_file_path):
    data_dir_name = os.path.dirname(zip_file_path)
    data_path = os.path.join(data_dir_name, "TissueNet")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)

    return data_path


def _load_images(data_path):
    list_of_split_paths = glob(os.path.join(data_path, "tissuenet*"))
    for split_path in list_of_split_paths:
        split_name = split_path.split("_")[-1].split(".")[0]

        z = np.load(split_path)
        x, y = z["X"], z["y"]
        for i, (im, label) in enumerate(zip(x, y)):
            _path = os.path.join(data_path, split_name, f"image_{i:04}.h5")
            with h5py.File(_path, "a") as f:
                f.create_dataset("raw/nucleus", data=im[0], compression="gzip")
                f.create_dataset("raw/cell", data=im[1], compression="gzip")
                f.create_dataset("labels/nucleus", data=label[0], compression="gzip")
                f.create_dataset("labels/cell", data=label[1], compression="gzip")


def get_tissuenet_loader(path, split, mode, download=False, **kwargs):
    assert split in ["train", "val", "test"]
    # TODO: integrate the conditions for getting the data, and doing the necessary parts (unzip, create data, etc)

    data_path = glob(os.path.join(path, split, "*.h5"))
    for _p in data_path:
        assert os.path.exists(_p)

    assert mode in ["nucleus", "cell"]
    raw_key, label_key = f"raw/{mode}", f"labels/{mode}"
    return torch_em.default_segmentation_loader(
        data_path, raw_key, data_path, label_key, **kwargs
    )
