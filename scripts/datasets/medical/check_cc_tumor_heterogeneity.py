import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.cc_tumor_heterogeneity import get_cc_tumor_heterogeneity_loader


sys.path.append("..")


def check_cc_tumor_heterogeneity():
    from util import ROOT

    loader = get_cc_tumor_heterogeneity_loader(
        path=os.path.join(ROOT, "cc_tumor_heterogeneity"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        ndim=2,
        resize_inputs=True,
        sampler=MinInstanceSampler(min_num_instances=3),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_cc_tumor_heterogeneity()
