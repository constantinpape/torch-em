import os
import sys

from torch_em.data.datasets import get_livecell_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_livecell():
    from util import ROOT

    livecell_root = os.path.join(ROOT, "livecell")
    loader = get_livecell_loader(livecell_root, "train", (512, 512), 1, download=True)
    check_loader(loader, 15, instance_labels=True)


if __name__ == "__main__":
    check_livecell()
