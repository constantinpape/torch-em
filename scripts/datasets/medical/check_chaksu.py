import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_chaksu_loader


sys.path.append("..")


def check_chaksu():
    from util import ROOT

    loader = get_chaksu_loader(
        path=os.path.join(ROOT, "chaksu"),
        patch_shape=(512, 512),
        batch_size=2,
        split="test",
        resize_inputs=True,
        task="optic_disc",
        download=True,
    )

    check_loader(loader, 8, rgb=True)


if __name__ == "__main__":
    check_chaksu()
