import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.light_microscopy.sperm_scd import get_sperm_scd_loader


sys.path.append("..")


def check_sperm_scd():
    from util import ROOT

    loader = get_sperm_scd_loader(
        path=os.path.join(ROOT, "sperm_scd"),
        batch_size=2,
        patch_shape=(512, 512),
        split="train",
        sampler=MinInstanceSampler(),
        download=True,
        shuffle=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="sperm_scd.png")


if __name__ == "__main__":
    check_sperm_scd()
