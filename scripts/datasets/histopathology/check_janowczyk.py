import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_janowczyk_loader


sys.path.append("..")


def check_janowczyk():
    from util import ROOT

    loader = get_janowczyk_loader(
        path=os.path.join(ROOT, "janowczyk"),
        patch_shape=(512, 512),
        annotation="nuclei",
        batch_size=2,
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_janowczyk()
