import os
import sys

from torch_em.data.datasets import get_fib25_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_fib25():
    from util import ROOT

    loader = get_fib25_loader(os.path.join(ROOT, "fib25"), (8, 512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./fib25.png")


if __name__ == "__main__":
    check_fib25()
