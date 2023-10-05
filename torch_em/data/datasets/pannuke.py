import os
import h5py
import vigra
import shutil
import numpy as np
from glob import glob

import torch_em
from torch_em.util.debug import check_loader
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

    is_fold_1 = os.path.exists(os.path.join(path, "pannuke_fold_1.h5"))
    is_fold_2 = os.path.exists(os.path.join(path, "pannuke_fold_2.h5"))
    is_fold_3 = os.path.exists(os.path.join(path, "pannuke_fold_3.h5"))
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


def _assort_pannuke_dataset(path):
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
        f.create_dataset(
            "images",
            data=np.load(os.path.join(path, "fold_1", "images", "fold1", "images.npy")).transpose(3, 0, 1, 2))
        f.create_dataset(
            "masks",
            data=np.load(os.path.join(path, "fold_1", "masks", "fold1", "masks.npy")).transpose(3, 0, 1, 2))

    # fold_2
    with h5py.File(h5_f2_path, "w") as f:
        f.create_dataset(
            "images",
            data=np.load(os.path.join(path, "fold_2", "images", "fold2", "images.npy")).transpose(3, 0, 1, 2))
        f.create_dataset(
            "masks",
            data=np.load(os.path.join(path, "fold_2", "masks", "fold2", "masks.npy")).transpose(3, 0, 1, 2))

    # fold_3
    with h5py.File(h5_f3_path, "w") as f:
        f.create_dataset(
            "images",
            data=np.load(os.path.join(path, "fold_3", "images", "fold3", "images.npy")).transpose(3, 0, 1, 2))
        f.create_dataset(
            "masks",
            data=np.load(os.path.join(path, "fold_3", "masks", "fold3", "masks.npy")).transpose(3, 0, 1, 2))

    dir_to_rm = glob(os.path.join(path, "*[!.h5]"))
    for tmp_dir in dir_to_rm:
        shutil.rmtree(tmp_dir)


def get_pannuke_dataset(
        path,
        patch_shape,
        folds=("fold_1", "fold_2", "fold_3"),
        rois={},
        download=False,
        with_channels=True,
        with_label_channels=True,
        **kwargs
):
    if rois is not None:
        assert isinstance(rois, dict)

    _download_pannuke_dataset(path, download)
    _convert_to_hdf5(path)

    data_paths = [os.path.join(path, f"pannuke_{f_idx}.h5") for f_idx in folds]
    data_rois = [rois.get(f_idx, np.s_[:, :, :]) for f_idx in folds]

    raw_key = "images"
    label_key = "masks"

    return torch_em.default_segmentation_dataset(
        data_paths, raw_key, data_paths, label_key, patch_shape, rois=data_rois,
        with_channels=with_channels, with_label_channels=with_label_channels, **kwargs
    )


def get_pannuke_loader(
        path,
        patch_shape,
        batch_size,
        folds=("fold_1", "fold_2", "fold_3"),
        download=False,
        rois={},
        **kwargs
):
    """TODO
    """
    dataset_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)

    ds = get_pannuke_dataset(
        path=path,
        patch_shape=patch_shape,
        folds=folds,
        rois=rois,
        download=download,
        **dataset_kwargs)
    return torch_em.get_data_loader(ds, batch_size=batch_size, **loader_kwargs)


def label_trafo(labels):
    """Converting the ground-truth of 6 (instance) channels into 1 label with instances from all channels
    channel info -
    (0: Neoplastic cells, 1: Inflammatory, 2: Connective/Soft tissue cells, 3: Dead Cells, 4: Epithelial, 6: Background)
    """
    max_labels_list = []
    f_labels = np.zeros_like(labels[1, :])
    for i, label in enumerate(labels):
        new_label, max_label, _ = vigra.analysis.relabelConsecutive(
            label.astype("uint64"),
            start_label=max_labels_list[-1] + 1 if len(max_labels_list) > 0 else 1)

        # some trailing channels might not have labels, hence appending only for elements with RoIs
        if max_label > 0:
            max_labels_list.append(max_label)

        # for the below written condition:
        # np.where(X, Y, Z), where,
        #   - X: condition "where" to look for
        #   - Y: we place 0 for the 6th channel (which has the "true bg" annotated as positive element), else we update it with the relabeled instances
        #   - Z: what to do outside the condition (mentioned in X)
        f_labels = np.where(new_label > 0, 0 if i == 5 else new_label, f_labels)

    return f_labels


def main():
    train_loader = get_pannuke_loader(
        path="./pannuke/",
        batch_size=2,
        patch_shape=(1, 256, 256),
        ndim=2,
        download=True,
        label_transform=label_trafo
    )
    check_loader(train_loader, 8, instance_labels=True, plt=False, rgb=True)


if __name__ == "__main__":
    main()
