import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_mif_tonsil_loader


sys.path.append("..")


def check_mif_tonsil():
    from util import ROOT

    for label_type in ["nuclei", "cells"]:
        loader = get_mif_tonsil_loader(
            path=os.path.join(ROOT, "mif_tonsil"),
            batch_size=2,
            patch_shape=(96, 96),
            label_type=label_type,
            download=True,
        )
        check_loader(loader, 4, instance_labels=True)


if __name__ == "__main__":
    check_mif_tonsil()
