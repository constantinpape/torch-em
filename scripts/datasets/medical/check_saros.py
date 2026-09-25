import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.saros import get_saros_loader


sys.path.append("..")


def check_saros():
    from util import ROOT

    loader = get_saros_loader(
        path=os.path.join(ROOT, "saros"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        label_type="regions",
        sampler=MinInstanceSampler(),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_saros()
