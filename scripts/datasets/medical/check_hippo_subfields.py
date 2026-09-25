import os
import sys

from torch_em.data import MinForegroundSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.hippo_subfields import get_hippo_subfields_loader


sys.path.append("..")


def check_hippo_subfields():
    from util import ROOT

    loader = get_hippo_subfields_loader(
        path=os.path.join(ROOT, "hippo_subfields"),
        patch_shape=(1, 256, 256),
        batch_size=2,
        ndim=2,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.001),
    )

    check_loader(loader, 8, plt=True, save_path="./hippo_subfields.png")


if __name__ == "__main__":
    check_hippo_subfields()
