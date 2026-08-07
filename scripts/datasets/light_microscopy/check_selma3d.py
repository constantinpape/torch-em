import os
import sys

from torch_em.data.datasets import get_selma3d_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_selma3d():
    from util import ROOT

    path = os.path.join(ROOT, "selma3d")
    for split, n_samples in (("train", 3), ("val", 1), ("test", 3)):
        loader = get_selma3d_loader(
            path=path,
            batch_size=1,
            patch_shape=(32, 128, 128),
            split=split,
            download=True,
        )
        check_loader(loader, n_samples, instance_labels=False)


if __name__ == "__main__":
    check_selma3d()
