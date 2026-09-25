import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.fetoplac import get_fetoplac_loader


sys.path.append("..")


def check_fetoplac():
    from util import ROOT

    loader = get_fetoplac_loader(
        path=os.path.join(ROOT, "fetoplac"),
        patch_shape=(384, 384),
        batch_size=2,
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./fetoplac.png")


if __name__ == "__main__":
    check_fetoplac()
