import os

import matplotlib
matplotlib.use("Agg")

from torch_em.data.datasets.electron_microscopy.probtem import get_probtem_loader  # noqa
from torch_em.util.debug import check_loader  # noqa

CIDAS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_probtem(split="train"):
    loader = get_probtem_loader(
        os.path.join(CIDAS_ROOT, "probtem"), batch_size=1, patch_shape=(512, 512),
        split=split, download=True,
    )
    check_loader(loader, 8, plt=True, save_path=f"./check_probtem_{split}.png")


def main():
    check_probtem("train")
    check_probtem("test")


if __name__ == "__main__":
    main()
