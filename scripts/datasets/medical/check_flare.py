import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.flare import get_flare_loader


sys.path.append("..")


def check_flare():
    from util import ROOT

    loader = get_flare_loader(
        path=os.path.join(ROOT, "flare"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        download=True,
        sampler=MinInstanceSampler(min_num_instances=3),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_flare()
