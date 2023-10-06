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
CHECKSUM = {
    "fold_1": "6e19ad380300e8ce9480f9ab6a14cc91fa4b6a511609b40e3d70bdf9c881ed0b",
    "fold_2": "5bc540cc509f64b5f5a274d6e5a245527dbd3e6d3155d43555115c5d54709b07",
    "fold_3": "c14d372981c42f611ebc80afad01702b89cad8c1b3089daa31931cf5a4b1a39d"
}


def _download_pannuke_dataset(path, download, folds):
    os.makedirs(path, exist_ok=True)

    checksum = CHECKSUM

    for tmp_fold in folds:
        if os.path.exists(os.path.join(path, f"pannuke_{tmp_fold}.h5")):
            return

        util.download_source(os.path.join(path, f"{tmp_fold}.zip"), URLS[tmp_fold], download, checksum[tmp_fold])

        print(f"Unzipping the PanNuke dataset in {tmp_fold} directories...")
        util.unzip(os.path.join(path, f"{tmp_fold}.zip"), os.path.join(path, f"{tmp_fold}"), True)


def _convert_to_hdf5(path, folds):
    for tmp_fold in folds:
        if os.path.exists(os.path.join(path, f"pannuke_{tmp_fold}.h5")):
            return

        print(f"Converting the {tmp_fold} into h5 file format...")
        tmp_name = tmp_fold.split("_")[0] + tmp_fold.split("_")[1]  # name of a particular sub-directory (per fold)
        img_path = glob(os.path.join(path, tmp_fold, "*", "images", tmp_name, "images.npy"))[0]
        gt_path = glob(os.path.join(path, tmp_fold, "*", "masks", tmp_name, "masks.npy"))[0]

        with h5py.File(os.path.join(path, f"pannuke_{tmp_fold}.h5"), "w") as f:
            f.create_dataset("images", data=np.load(img_path).transpose(3, 0, 1, 2))
            f.create_dataset("masks", data=np.load(gt_path).transpose(3, 0, 1, 2))

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

    _download_pannuke_dataset(path, download, folds)
    _convert_to_hdf5(path, folds)

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
    segmentation = np.zeros(labels.shape[1:])
    max_ids = []
    for label_channel in labels[:-1]:
        # the 'start_label' takes care of where to start allocating the instance ids from
        this_labels, max_id, _ = vigra.analysis.relabelConsecutive(
            label_channel.astype("uint64"),
            start_label=max_ids[-1] + 1 if len(max_ids) > 0 else 1)

        # some trailing channels might not have labels, hence appending only for elements with RoIs
        if max_id > 0:
            max_ids.append(max_id)

        segmentation[this_labels > 0] = this_labels[this_labels > 0]

    return segmentation


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
