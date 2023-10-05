import os
import h5py
import shutil
import numpy as np
from glob import glob

from torch_em.data.datasets import util


# PanNuke Dataset - https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
URLS = {
    "fold_1": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_1.zip",
    "fold_2": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_2.zip",
    "fold_3": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_3.zip"
}


# TODO
CHECKSUM = None


def _download_pannuke_dataset(path, download):
    os.makedirs(path, exist_ok=True)

    is_fold_1 = os.path.exists(os.path.join(path, "fold_1"))
    is_fold_2 = os.path.exists(os.path.join(path, "fold_2"))
    is_fold_3 = os.path.exists(os.path.join(path, "fold_3"))
    if is_fold_1 and is_fold_2 and is_fold_3 is True:
        return

    checksum = CHECKSUM
    util.download_source(os.path.join(path, "fold_1.zip"), URLS["fold_1"], download, checksum)
    util.download_source(os.path.join(path, "fold_2.zip"), URLS["fold_2"], download, checksum)
    util.download_source(os.path.join(path, "fold_3.zip"), URLS["fold_3"], download, checksum)

    print("Unzipping the PanNuke dataset in their respective fold directories...")
    util.unzip(os.path.join(path, "fold_1.zip"), os.path.join(path, "fold_1"), True)
    util.unzip(os.path.join(path, "fold_2.zip"), os.path.join(path, "fold_2"), True)
    util.unzip(os.path.join(path, "fold_3.zip"), os.path.join(path, "fold_3"), True)

    _assort_pannuke_dataset(path, download)


def _assort_pannuke_dataset(path, download):
    if download:
        print("Assorting the PanNuke dataset in the expected structure...")
        for _p in glob(os.path.join(path, "*", "*")):
            dst = os.path.split(_p)[0]
            for src_sub_dir in glob(os.path.join(_p, "*")):
                f_dst = os.path.join(dst, os.path.split(src_sub_dir)[-1])
                shutil.move(src_sub_dir, f_dst)
            shutil.rmtree(_p)


def _convert_to_hdf5(path):
    h5_f1_path = os.path.join(path, "pannuke_fold_1.h5")
    h5_f2_path = os.path.join(path, "pannuke_fold_2.h5")
    h5_f3_path = os.path.join(path, "pannuke_fold_3.h5")

    if os.path.exists(h5_f1_path) and os.path.exists(h5_f2_path) and os.path.exists(h5_f3_path) is True:
        return

    print("Converting the folds into h5 file format...")
    # fold_1
    with h5py.File(h5_f1_path, "w") as f:
        f.create_dataset("images", data=np.load(os.path.join(path, "fold_1", "images", "fold1", "images.npy")))
        f.create_dataset("masks", data=np.load(os.path.join(path, "fold_1", "masks", "fold1", "masks.npy")))

    # fold_2
    with h5py.File(h5_f2_path, "w") as f:
        f.create_dataset("images", data=np.load(os.path.join(path, "fold_2", "images", "fold2", "images.npy")))
        f.create_dataset("masks", data=np.load(os.path.join(path, "fold_2", "masks", "fold2", "masks.npy")))

    # fold_3
    with h5py.File(h5_f3_path, "w") as f:
        f.create_dataset("images", data=np.load(os.path.join(path, "fold_3", "images", "fold3", "images.npy")))
        f.create_dataset("masks", data=np.load(os.path.join(path, "fold_3", "masks", "fold3", "masks.npy")))

    dir_to_rm = glob(os.path.join(path, "*[!.h5]"))
    for tmp_dir in dir_to_rm:
        shutil.rmtree(tmp_dir)


def get_pannuke_dataset(path, download):
    _download_pannuke_dataset(path, download)
    _convert_to_hdf5(path)


def get_pannuke_loader(path, download):
    get_pannuke_dataset(path, download)


def main():
    path = "./pannuke/"
    download = True

    get_pannuke_loader(path, download)


if __name__ == "__main__":
    main()
