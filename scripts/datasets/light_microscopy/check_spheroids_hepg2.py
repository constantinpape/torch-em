import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_spheroids_hepg2_loader


sys.path.append("..")


def check_spheroids_hepg2():
    from util import ROOT

    loader = get_spheroids_hepg2_loader(
        path=os.path.join(ROOT, "spheroids_hepg2"),
        batch_size=1,
        patch_shape=(32, 256, 256),
        download=True,
        sampler=MinInstanceSampler()
    )

    check_loader(loader, 4, instance_labels=True, plt=True, save_path="spheroids_hepg2.png")


if __name__ == "__main__":
    check_spheroids_hepg2()
