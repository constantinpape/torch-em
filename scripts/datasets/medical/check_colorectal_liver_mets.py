import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.colorectal_liver_mets import get_colorectal_liver_mets_loader


sys.path.append("..")


def check_colorectal_liver_mets():
    from util import ROOT

    loader = get_colorectal_liver_mets_loader(
        path=os.path.join(ROOT, "colorectal_liver_mets"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        sampler=MinInstanceSampler(min_num_instances=3),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_colorectal_liver_mets()
