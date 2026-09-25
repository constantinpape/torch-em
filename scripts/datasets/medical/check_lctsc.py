import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.lctsc import get_lctsc_loader


sys.path.append("..")


def check_lctsc():
    from util import ROOT

    loader = get_lctsc_loader(
        path=os.path.join(ROOT, "lctsc"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        sampler=MinInstanceSampler(min_num_instances=3),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_lctsc()
