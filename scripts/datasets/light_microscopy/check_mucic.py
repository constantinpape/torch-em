import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_mucic_loader


sys.path.append("..")


def check_mucic_colon_tissue():
    from util import ROOT

    loader = get_mucic_loader(
        path=os.path.join(ROOT, "mucic"),
        batch_size=1,
        patch_shape=(16, 256, 256),
        cell_line="colon_tissue",
        variant="low",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


def check_mucic_hl60():
    from util import ROOT

    loader = get_mucic_loader(
        path=os.path.join(ROOT, "mucic"),
        batch_size=1,
        patch_shape=(16, 256, 256),
        cell_line="hl60",
        variant="low_c00",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


def check_mucic_granulocytes():
    from util import ROOT

    loader = get_mucic_loader(
        path=os.path.join(ROOT, "mucic"),
        batch_size=1,
        patch_shape=(16, 256, 256),
        cell_line="granulocytes",
        variant="low",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


def check_mucic_vasculogenesis():
    from util import ROOT

    loader = get_mucic_loader(
        path=os.path.join(ROOT, "mucic"),
        batch_size=1,
        patch_shape=(256, 256),
        cell_line="vasculogenesis",
        variant="default",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    # check_mucic_colon_tissue()
    # check_mucic_hl60()
    # check_mucic_granulocytes()
    check_mucic_vasculogenesis()
