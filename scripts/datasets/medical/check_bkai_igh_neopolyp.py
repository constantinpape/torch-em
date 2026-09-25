import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_bkai_igh_neopolyp_loader


sys.path.append("..")


def check_bkai_igh_neopolyp():
    from util import ROOT

    loader = get_bkai_igh_neopolyp_loader(
        path=os.path.join(ROOT, "bkai_igh_neopolyp"),
        patch_shape=(512, 512),
        batch_size=2,
        download=True,
        resize_inputs=False,
    )

    check_loader(loader, 8, plt=True, save_path="./bkai_igh_neopolyp.png")


if __name__ == "__main__":
    check_bkai_igh_neopolyp()
