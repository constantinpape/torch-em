import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_celegans_atlas_loader


sys.path.append("..")


def check_celegans_atlas():
    from util import ROOT

    loader = get_celegans_atlas_loader(
        path=os.path.join(ROOT, "celegans_atlas"),
        batch_size=2,
        patch_shape=(8, 256, 256),
        download=True,
        split="train",
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_celegans_atlas()
