import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_cochleanet_loader


sys.path.append("..")


def check_cochleanet():
    from util import ROOT

    for target in ("sgn", "ihc"):
        loader = get_cochleanet_loader(
            path=os.path.join(ROOT, "cochleanet"),
            batch_size=1,
            patch_shape=(64, 128, 128),
            split="train",
            target=target,
            sampler=MinInstanceSampler(),
            download=True,
        )
        check_loader(loader, 4, instance_labels=True, plt=True, save_path=f"cochleanet_{target}.png")


if __name__ == "__main__":
    check_cochleanet()
