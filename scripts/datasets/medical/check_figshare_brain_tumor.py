import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.figshare_brain_tumor import get_figshare_brain_tumor_loader


sys.path.append("..")


def check_figshare_brain_tumor():
    from util import ROOT

    loader = get_figshare_brain_tumor_loader(
        path=os.path.join(ROOT, "figshare_brain_tumor"),
        patch_shape=(512, 512),
        batch_size=1,
        sampler=MinInstanceSampler(),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_figshare_brain_tumor()
