import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.imagecas import get_imagecas_loader


sys.path.append("..")


def check_imagecas():
    from util import ROOT

    loader = get_imagecas_loader(
        path=os.path.join(ROOT, "imagecas"),
        patch_shape=(32, 512, 512),
        batch_size=1,
        sampler=MinInstanceSampler(),
        ndim=3,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_imagecas()
