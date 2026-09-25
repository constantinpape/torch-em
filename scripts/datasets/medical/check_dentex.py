import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.dentex import get_dentex_loader


sys.path.append("..")


def check_dentex():
    from util import ROOT

    loader = get_dentex_loader(
        path=os.path.join(ROOT, "dentex"),
        patch_shape=(512, 512),
        batch_size=2,
        split="val",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./dentex.png")


if __name__ == "__main__":
    check_dentex()
