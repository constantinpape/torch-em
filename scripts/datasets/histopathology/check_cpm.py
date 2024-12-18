import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_cpm_loader


sys.path.append("..")


def check_cpm():
    from util import ROOT

    loader = get_cpm_loader(
        path=os.path.join(ROOT, "cpm"),
        patch_shape=(512, 512),
        batch_size=2,
        data_choice="cpm17",
        split="train",
    )
    check_loader(loader, 8, rgb=True, instance_labels=True)


if __name__ == "__main__":
    check_cpm()
