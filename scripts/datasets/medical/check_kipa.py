import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.kipa import get_kipa_loader


sys.path.append("..")


def check_kipa():
    from util import ROOT

    loader = get_kipa_loader(
        path=os.path.join(ROOT, "kipa"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_kipa()
