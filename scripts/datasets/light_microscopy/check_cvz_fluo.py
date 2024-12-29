import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_cvz_fluo_loader


sys.path.append("..")


def check_cvz_fluo():
    from util import ROOT

    loader = get_cvz_fluo_loader(
        path=os.path.join(ROOT, "cvz"),
        patch_shape=(512, 512),
        batch_size=2,
        stain_choice="cell",
        data_choice=None,
    )

    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./test.png", rgb=True)


if __name__ == "__main__":
    check_cvz_fluo()
