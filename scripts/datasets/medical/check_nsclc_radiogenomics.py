import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.nsclc_radiogenomics import get_nsclc_radiogenomics_loader


sys.path.append("..")


def check_nsclc_radiogenomics():
    from util import ROOT

    loader = get_nsclc_radiogenomics_loader(
        path=os.path.join(ROOT, "nsclc_radiogenomics"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        sampler=MinInstanceSampler(min_num_instances=2),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_nsclc_radiogenomics()
