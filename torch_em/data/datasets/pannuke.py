import os
import shutil
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

    if os.path.exists(os.path.join(path, "images")) and os.path.exists(os.path.join(path, "annotations")) is True:
        return

    checksum = CHECKSUM
    util.download_source(os.path.join(path, "fold_1.zip"), URLS["fold_1"], download, checksum)
    util.download_source(os.path.join(path, "fold_2.zip"), URLS["fold_2"], download, checksum)
    util.download_source(os.path.join(path, "fold_3.zip"), URLS["fold_3"], download, checksum)

    print("Unzipping the PanNuke dataset in their respective fold directories")
    util.unzip(os.path.join(path, "fold_1.zip"), os.path.join(path, "fold_1"), True)
    util.unzip(os.path.join(path, "fold_2.zip"), os.path.join(path, "fold_2"), True)
    util.unzip(os.path.join(path, "fold_3.zip"), os.path.join(path, "fold_3"), True)

    _assort_pannuke_dataset(path, download)


def _assort_pannuke_dataset(path, download):
    if download:
        print("Assorting the PanNuke dataset in the expected structure")
        for _p in glob(os.path.join(path, "*", "*")):
            dst = os.path.split(_p)[0]
            for src_sub_dir in glob(os.path.join(_p, "*")):
                f_dst = os.path.join(dst, os.path.split(src_sub_dir)[-1])
                shutil.move(src_sub_dir, f_dst)
            shutil.rmtree(_p)

        # sorting the respective fold's:
        #   - "path/fold_<ID>/images" and "path/fold_<ID>/masks" -> "path/images" and "path/annotations"
        #   - (inspired from LiveCell data structure)
        img_dir = os.path.join(path, "images")
        gt_dir = os.path.join(path, "annotations")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        for _p in glob(os.path.join(path, "*")):
            fold_name = os.path.split(_p)[-1]
            for sub_dir in glob(os.path.join(_p, "*")):
                new_dir = os.path.join(gt_dir if os.path.split(sub_dir)[-1] == "masks" else img_dir,
                                       fold_name, os.path.split(sub_dir)[-1])
                shutil.move(sub_dir, new_dir)
            shutil.rmtree(_p)


def main():
    path = "/scratch/usr/nimanwai/tmp_test/"
    download = True

    _download_pannuke_dataset(path, download)


if __name__ == "__main__":
    main()
