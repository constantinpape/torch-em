import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_parhyale_regen_loader


sys.path.append("..")


def check_parhyale_regen():
    from util import ROOT

    loader = get_parhyale_regen_loader(
        path=os.path.join(ROOT, "parhyale_regen"),
        batch_size=2,
        patch_shape=(8, 256, 256),
        download=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_parhyale_regen()
