import os

import numpy as np

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import light_microscopy, electron_microscopy


ROOT = "/scratch/projects/nim00007/sam/data"
# ROOT = "/media/anwai/ANWAI/data/"


def _fetch_loaders(dataset_name):
    if dataset_name == "covid_if":
        # 1, Covid IF does not have internal splits. For this example I chose first 10 samples for training,
        # and next 3 samples for validation, left the rest for testing.
        train_loader = light_microscopy.get_covid_if_loader(
            path=os.path.join(ROOT, "covid_if"),
            patch_shape=(512, 512),
            batch_size=2,
            sample_range=(None, 10),
            target="cells",
            num_workers=16,
            shuffle=True,
            download=True,
        )
        val_loader = light_microscopy.get_covid_if_loader(
            path=os.path.join(ROOT, "covid_if"),
            patch_shape=(512, 512),
            batch_size=1,
            sample_range=(10, 13),
            target="cells",
            num_workers=16,
            download=True,
        )

    elif dataset_name == "orgasegment":
        # 2. OrgaSegment has internal splits provided. We follow the respective splits for our experiments.
        train_loader = light_microscopy.get_orgasegment_loader(
            path=os.path.join(ROOT, "orgasegment"),
            patch_shape=(512, 512),
            split="train",
            batch_size=2,
            num_workers=16,
            shuffle=True,
            download=True,
        )
        val_loader = light_microscopy.get_orgasegment_loader(
            path=os.path.join(ROOT, "orgasegment"),
            patch_shape=(512, 512),
            split="val",
            batch_size=1,
            num_workers=16,
            download=True,
        )

    elif dataset_name == "mouse-embryo":
        # 3. Mouse Embryo
        # the logic used here is: I use the first 100 slices per volume from the training split for training
        # and the next ~20/30 slices per volume from the training split for validation
        # and we use the whole volume from the val set for testing
        train_rois = [np.s_[0:100, :, :], np.s_[0:100, :, :], np.s_[0:100, :, :], np.s_[0:100, :, :]]
        val_rois = [np.s_[100:, :, :], np.s_[100:, :, :], np.s_[100:, :, :], np.s_[100:, :, :]]

        train_loader = light_microscopy.get_mouse_embryo_loader(
            path=os.path.join(ROOT, "mouse-embryo"),
            name="membrane",
            split="train",
            patch_shape=(1, 512, 512),
            batch_size=1,
            download=True,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(min_num_instances=3),
            rois=train_rois,
        )
        val_loader = light_microscopy.get_mouse_embryo_loader(
            path=os.path.join(ROOT, "mouse-embryo"),
            name="membrane",
            split="train",
            patch_shape=(1, 512, 512),
            batch_size=1,
            download=True,
            num_workers=16,
            sampler=MinInstanceSampler(min_num_instances=3),
            rois=val_rois,
        )

    elif dataset_name == "mitolab_glycolytic_muscle":
        # 4. This dataset would need aspera-cli to be installed, I'll provide you with this data
        # ...
        train_rois = np.s_[0:175, :, :]
        val_rois = np.s_[175:225, :, :]
        test_rois = np.s_[225:, :, :]
        train_loader = electron_microscopy.cem.get_benchmark_loader(
            path=os.path.join(ROOT, "mitolab"),
            dataset_id=3,
            batch_size=2,
            patch_shape=(1, 512, 512),
            download=False,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(),
            rois=train_rois,
        )
        val_loader = electron_microscopy.cem.get_benchmark_loader(
            path=os.path.join(ROOT, "mitolab"),
            dataset_id=3,
            batch_size=2,
            patch_shape=(1, 512, 512),
            download=False,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(),
            rois=val_rois,
        )

    elif dataset_name == "platy_cilia":
        # 5. Platynereis (Cilia)
        # the logic used here is: I use the first 85 slices per volume from the training split for training
        # and the next ~10-15 slices per volume from the training split for validation
        # and we use the whole volume from the val set for testing
        train_rois = {
            1: np.s_[0:85, :, :], 2: np.s_[0:85, :, :], 3: np.s_[0:85, :, :]
        }
        val_rois = {
            1: np.s_[85:, :, :], 2: np.s_[85:, :, :], 3: np.s_[85:, :, :]
        }

        train_loader = electron_microscopy.get_platynereis_cilia_loader(
            path=os.path.join(ROOT, "platynereis"),
            patch_shape=(1, 512, 512),
            ndim=2,
            batch_size=2,
            rois=train_rois,
            download=True,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(),
        )
        val_loader = electron_microscopy.get_platynereis_cilia_loader(
            path=os.path.join(ROOT, "platynereis"),
            patch_shape=(1, 512, 512),
            ndim=2,
            batch_size=1,
            rois=val_rois,
            download=True,
            num_workers=16,
            sampler=MinInstanceSampler(),
        )

    else:
        raise ValueError(f"{dataset_name} is not a valid dataset name.")

    return train_loader, val_loader


def _verify_loaders():
    dataset_name = "mitolab_glycolytic_muscle"

    train_loader, val_loader = _fetch_loaders(dataset_name=dataset_name)

    breakpoint()

    # NOTE: if using on the cluster, napari visualization won't work with "check_loader".
    # turn "plt=True" and provide path to save the matplotlib outputs of the loader.
    check_loader(train_loader, 8, plt=True, save_path="./train_loader.png")
    check_loader(val_loader, 8, plt=True, save_path="./val_loader.png")


if __name__ == "__main__":
    _verify_loaders()
