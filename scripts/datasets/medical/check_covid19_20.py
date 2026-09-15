import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.covid19_20 import get_covid19_20_loader


sys.path.append("..")


def check_covid19_20():
    from util import ROOT

    loader = get_covid19_20_loader(
        path=os.path.join(ROOT, "covid19_20"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        download=True,
        sampler=MinInstanceSampler(min_num_instances=2),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_covid19_20()
