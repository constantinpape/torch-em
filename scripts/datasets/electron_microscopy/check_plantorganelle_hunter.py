import os
import sys

from torch_em.data.datasets import get_plantorganelle_hunter_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_plantorganelle_hunter():
    from util import ROOT

    loader = get_plantorganelle_hunter_loader(
        path=os.path.join(ROOT, "plantorganelle_hunter"),
        batch_size=1,
        patch_shape=(512, 512),
        split="train",
        label_choice=None,
        download=True,
    )
    check_loader(loader, 8, instance_labels=False)


if __name__ == "__main__":
    check_plantorganelle_hunter()
