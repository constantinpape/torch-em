import os
import sys

sys.path.append("..")


def check_microns_loader():
    from torch_em.data.datasets.electron_microscopy import get_microns_loader
    from torch_em.util.debug import check_loader
    from util import ROOT

    loader = get_microns_loader(
        os.path.join(ROOT, "microns"), batch_size=1, patch_shape=(8, 512, 512), download=True
    )
    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./check_microns.png")


def check_microns_mito_loader():
    from torch_em.data.datasets.electron_microscopy import get_microns_loader
    from torch_em.util.debug import check_loader
    from util import ROOT

    loader = get_microns_loader(
        os.path.join(ROOT, "microns"), batch_size=1, patch_shape=(8, 512, 512),
        label_choice="mitochondria", download=True
    )
    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./check_microns_mito.png")


def check_microns_minnie65_loader():
    from torch_em.data.datasets.electron_microscopy import get_microns_minnie65_loader
    from torch_em.util.debug import check_loader
    from util import ROOT

    loader = get_microns_minnie65_loader(
        os.path.join(ROOT, "microns-minnie65"), batch_size=1, patch_shape=(8, 1024, 1024),
        split="train", download=True
    )
    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./check_microns_minnie65.png")


def main():
    check_microns_loader()
    check_microns_mito_loader()
    check_microns_minnie65_loader()


if __name__ == "__main__":
    main()
