import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_curvas_loader


sys.path.append("..")


def check_curvas():
    # from util import ROOT
    ROOT = "/scratch/share/cidas/cca/data"

    loader = get_curvas_loader(
        path=os.path.join(ROOT, "curvas"),
        patch_shape=(1, 512, 512),
        batch_size=2,
        ndim=2,
        rater="1",
        resize_inputs=False,
        download=True,
        sampler=MinInstanceSampler()
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_curvas()
