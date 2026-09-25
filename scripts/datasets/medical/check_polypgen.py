import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.polypgen import get_polypgen_loader


sys.path.append("..")


def check_polypgen():
    from util import ROOT

    loader = get_polypgen_loader(
        path=os.path.join(ROOT, "polypgen"),
        patch_shape=(512, 512),
        batch_size=1,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./polypgen.png")


if __name__ == "__main__":
    check_polypgen()
