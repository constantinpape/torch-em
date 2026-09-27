import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_prostate_cnb_loader


sys.path.append("..")


def check_prostate_cnb():
    from util import ROOT

    loader = get_prostate_cnb_loader(
        path=os.path.join(ROOT, "prostate_cnb"),
        batch_size=1,
        patch_shape=(512, 512),
        n_cases=2,
        level=1,
        download=True,
    )

    check_loader(loader, 8, instance_labels=False, rgb=True, plt=True, save_path="prostate_cnb.png")


if __name__ == "__main__":
    check_prostate_cnb()
