import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_sinus_loader


sys.path.append("..")


def check_sinus():
    from util import ROOT

    loader = get_sinus_loader(
        path=os.path.join(ROOT, "sinus"),
        patch_shape=(512, 512),
        batch_size=1,
        annotation_choice="inclusive",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True, rgb=True)


if __name__ == "__main__":
    check_sinus()
