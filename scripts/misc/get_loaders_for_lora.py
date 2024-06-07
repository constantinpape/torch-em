import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import light_microscopy, electron_microscopy


ROOT = "/media/anwai/ANWAI/data/"


def _fetch_loaders(dataset_name):
    if dataset_name == "covid_if":
        # 1, Covid IF does ot have internal splits. For this example I chose first 10 samples for training,
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

    elif dataset_name == "mouse_embryo":
        # 3. Mouse Embryo
        # TODO: @AA: one particular volume seens to have annotations for the bg, need to investigate this.
        # TODO: make roi splits in favor of using "val" for testing purposes.
        train_loader = light_microscopy.get_mouse_embryo_loader(
            path=os.path.join(ROOT, "mouse_embryo"),
            name="membrane",
            split="train",
            patch_shape=(1, 512, 512),
            batch_size=1,
            download=True,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(min_num_instances=3)
        )
        val_loader = light_microscopy.get_mouse_embryo_loader(
            path=os.path.join(ROOT, "mouse_embryo"),
            name="membrane",
            split="train",
            patch_shape=(1, 512, 512),
            batch_size=1,
            download=True,
            num_workers=16,
            sampler=MinInstanceSampler(min_num_instances=3)
        )

    elif dataset_name == "mitolab_glycotic_muscle":
        # 4. This dataset would need aspera-cli to be installed, I'll provide you with this data
        # TODO: @AA: test this and check this one out.
        loader = electron_microscopy.cem.get_benchmark_loader(
            path=os.path.join(ROOT, "mitolab"),
            dataset_id=2,
            batch_size=2,
            patch_shape=(1, 512, 512),
            download=True,
            num_workers=16,
            shuffle=True,
        )
        train_loader = loader
        val_loader = loader

    elif dataset_name == "platy_cilia":
        # 5. Platynereis (Cilia)
        # TODO: @AA: check this
        loader = electron_microscopy.get_platynereis_cilia_loader(
            path=os.path.join(ROOT, "platynereis"),
            patch_shape=(1, 512, 512),
            ndim=2,
            batch_size=2,
            sample_ids=...,
            rois=...,
            download=True,
            num_workers=16,
            shuffle=True,
        )

    return train_loader, val_loader


def _verify_loaders():
    dataset_name = "mouse_embryo"

    train_loader, val_loader = _fetch_loaders(dataset_name=dataset_name)

    # NOTE: if using on the cluster, napari visualization won't work with "check_loader".
    # turn "plt=True" and provide path to save the matplotlib outputs of the loader.
    check_loader(train_loader, 8)
    check_loader(val_loader, 8)


if __name__ == "__main__":
    _verify_loaders()
