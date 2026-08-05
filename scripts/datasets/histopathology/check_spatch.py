import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_spatch_loader


sys.path.append("..")


def check_spatch():
    from util import ROOT

    for subset in ["xenium_ov", "stereoseq_ov"]:
        loader = get_spatch_loader(
            path=os.path.join(ROOT, "spatch"),
            patch_shape=(512, 512),
            batch_size=2,
            subset=subset,
            download=True,
            shuffle=True,
        )

        check_loader(loader, 8, instance_labels=True, rgb=True)


if __name__ == "__main__":
    check_spatch()
