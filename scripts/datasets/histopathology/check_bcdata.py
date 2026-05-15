import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_bcdata_loader


sys.path.append("..")


def check_bcdata(binary):
    from util import ROOT

    loader = get_bcdata_loader(
        path=os.path.join(ROOT, "bcdata"),
        split="train",
        patch_shape=(512, 512),
        batch_size=2,
        cell_radius=2,
        binary=binary,
        download=True,
    )

    print(f"BCData | binary={binary} | #batches={len(loader)}")
    check_loader(loader, 8, instance_labels=False, rgb=True)


if __name__ == "__main__":
    check_bcdata(binary=False)
    check_bcdata(binary=True)
