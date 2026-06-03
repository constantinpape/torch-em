import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_vibrio_cholerae_loader


sys.path.append("..")


def check_vibrio_cholerae():
    from util import ROOT

    loader = get_vibrio_cholerae_loader(
        path=os.path.join(ROOT, "vibrio_cholerae"),
        batch_size=1,
        patch_shape=(32, 256, 256),
        download=True,
        sampler=MinInstanceSampler()
    )

    check_loader(loader, 4, instance_labels=True, plt=True, save_path="vibrio_cholerae.png")


if __name__ == "__main__":
    check_vibrio_cholerae()
