import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_camelyon16_loader, get_camelyon17_loader


sys.path.append("..")


def check_camelyon16():
    from util import ROOT

    loader = get_camelyon16_loader(
        path=os.path.join(ROOT, "camelyon16"),
        patch_shape=(512, 512),
        batch_size=1,
        sample_ids=["tumor_091", "normal_108"],
        resolution_level=3,
        download=True,
    )

    check_loader(loader, 8, instance_labels=False, rgb=True)


def check_camelyon17():
    from util import ROOT

    loader = get_camelyon17_loader(
        path=os.path.join(ROOT, "camelyon17"),
        patch_shape=(512, 512),
        batch_size=1,
        resolution_level=3,
        download=True,
    )

    check_loader(loader, 8, instance_labels=False, rgb=True)


if __name__ == "__main__":
    check_camelyon16()
    check_camelyon17()
