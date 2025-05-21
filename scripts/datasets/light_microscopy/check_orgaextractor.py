import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_orgaextractor_loader


sys.path.append("..")


def check_orgaextractor():
    from util import ROOT

    loader = get_orgaextractor_loader(
        path=os.path.join(ROOT, "orgaextractor"),
        batch_size=2,
        patch_shape=(512, 512),
        split="test",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_orgaextractor()
