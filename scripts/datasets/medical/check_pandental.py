import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.pandental import get_pandental_loader


sys.path.append("..")


def check_pandental():
    from util import ROOT

    loader = get_pandental_loader(
        path=os.path.join(ROOT, "pandental"),
        patch_shape=(512, 512),
        batch_size=2,
        annotator="1",
        resize_inputs=False,
        download=True,
    )
    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_pandental()
