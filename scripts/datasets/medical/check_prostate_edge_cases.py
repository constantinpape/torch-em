import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.prostate_edge_cases import get_prostate_edge_cases_loader


sys.path.append("..")


def check_prostate_edge_cases():
    from util import ROOT

    loader = get_prostate_edge_cases_loader(
        path=os.path.join(ROOT, "prostate_edge_cases"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_prostate_edge_cases()
