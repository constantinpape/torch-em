import os
import sys

from torch_em.data.datasets import get_lizard_mitosis_loader
from torch_em.util.debug import check_loader


sys.path.append("..")


def check_lizard_mitosis():
    from util import ROOT

    loader = get_lizard_mitosis_loader(
        path=os.path.join(ROOT, "lizard_mitosis"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        subset="lizard",
        split="val",
        download=True,
    )
    check_loader(loader, 8, rgb=True, instance_labels=True)


if __name__ == "__main__":
    check_lizard_mitosis()
