import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_organoid_loader


sys.path.append("..")


def check_organoid():
    from util import ROOT

    loader = get_organoid_loader(
        path=os.path.join(ROOT, "organoid"),
        batch_size=2,
        patch_shape=(8, 512, 512),
        source="gemcitabine",
        source_channels="bf",
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_organoid()
