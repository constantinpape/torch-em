import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_hest_loader


sys.path.append("..")


def check_hest():
    from util import ROOT

    loader = get_hest_loader(
        path=os.path.join(ROOT, "hest"),
        patch_shape=(224, 224),
        batch_size=2,
        organs=["Lung"],
        label_choice="instances",
        download=True,
    )
    check_loader(loader, 8, instance_labels=True, rgb=True)


if __name__ == "__main__":
    check_hest()
