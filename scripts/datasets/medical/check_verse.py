import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_verse_loader


sys.path.append("..")


def check_verse():
    from util import ROOT

    loader = get_verse_loader(
        path=os.path.join(ROOT, "verse"),
        split="test",
        patch_shape=(1, 512, 512),
        batch_size=2,
        download=True,
        resize_inputs=False,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_verse()
