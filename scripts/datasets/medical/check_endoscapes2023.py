import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_endoscapes2023_loader


sys.path.append("..")


def check_endoscapes2023():
    from util import ROOT

    loader = get_endoscapes2023_loader(
        path=os.path.join(ROOT, "endoscapes2023"),
        split="train",
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, rgb=True)


if __name__ == "__main__":
    check_endoscapes2023()
