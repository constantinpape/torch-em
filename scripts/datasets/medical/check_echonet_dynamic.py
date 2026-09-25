import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.echonet_dynamic import get_echonet_dynamic_loader


sys.path.append("..")


def check_echonet_dynamic():
    from util import ROOT

    loader = get_echonet_dynamic_loader(
        path=os.path.join(ROOT, "echonet_dynamic"),
        patch_shape=(1, 112, 112),
        batch_size=1,
        sampler=MinInstanceSampler(),
        ndim=2,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_echonet_dynamic()
