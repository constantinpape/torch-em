import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.ct_lymph_nodes import get_ct_lymph_nodes_loader


sys.path.append("..")


def check_ct_lymph_nodes():
    from util import ROOT

    loader = get_ct_lymph_nodes_loader(
        path=os.path.join(ROOT, "ct_lymph_nodes"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_ct_lymph_nodes()
