import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_motum_loader


sys.path.append("..")


def check_motum():
    from util import ROOT

    loader = get_motum_loader(
        path=os.path.join(ROOT, "motum"),
        batch_size=1,
        patch_shape=(8, 512, 512),
        ndim=3,
        resize_inputs=True,
        modality="t1ce",
        download=False,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_motum()
