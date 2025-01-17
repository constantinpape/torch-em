import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_monusac_loader


sys.path.append("..")


def check_monusac():
    from util import ROOT

    test_loader = get_monusac_loader(
        path=os.path.join(ROOT, "monusac"),
        download=True,
        patch_shape=(512, 512),
        batch_size=1,
        split="test",
        organ_type=None,
    )
    print("Length of test loader: ", len(test_loader))
    check_loader(test_loader, 8, instance_labels=True, rgb=True)


if __name__ == "__main__":
    check_monusac()
