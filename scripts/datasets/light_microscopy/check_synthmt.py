import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_synthmt_loader


sys.path.append("..")


def check_synthmt(source, split):
    from util import ROOT

    loader = get_synthmt_loader(
        path=os.path.join(ROOT, "synthmt"),
        batch_size=1,
        patch_shape=(512, 512),
        source=source,
        split=split,
        download=True,
    )

    check_loader(loader, 8, instance_labels=source == "synthetic_irm", rgb=source == "synthetic_irm")


def main():
    check_synthmt("synthetic_irm", None)
    check_synthmt("real_irm", "train")
    check_synthmt("real_irm", "test")


if __name__ == "__main__":
    main()
