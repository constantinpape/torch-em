import os

import matplotlib
matplotlib.use("Agg")

from torch_em.data.datasets.electron_microscopy.deepcontact import get_deepcontact_loader  # noqa
from torch_em.util.debug import check_loader  # noqa

CIDAS_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_deepcontact(label_choice="mito", sources=None):
    loader = get_deepcontact_loader(
        os.path.join(CIDAS_ROOT, "deepcontact"), batch_size=1, patch_shape=(512, 512),
        label_choice=label_choice, sources=sources, download=True,
    )
    check_loader(loader, 8, plt=True, save_path=f"./check_deepcontact_{label_choice}.png")


def main():
    check_deepcontact("mito")
    check_deepcontact("er")
    check_deepcontact("ld")


if __name__ == "__main__":
    main()
