import os
import sys


from torch_em.data.datasets import get_vnc_mito_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_vnc():
    from util import ROOT

    loader = get_vnc_mito_loader(os.path.join(ROOT, "vnc"), (8, 512, 512), 1, download=True)
    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_vnc()
