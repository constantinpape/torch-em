import os
import sys

from torch_em.data.datasets import get_e11bio_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_e11bio():
    from util import ROOT

    loader = get_e11bio_loader(
        path=os.path.join(ROOT, "e11bio"),
        patch_shape=(32, 256, 256),
        batch_size=1,
        split="instance",
        crop_ids=list(range(5, 10)),
        channel=0,
        download=True,
    )
    check_loader(loader, 4, instance_labels=True)


if __name__ == "__main__":
    check_e11bio()
