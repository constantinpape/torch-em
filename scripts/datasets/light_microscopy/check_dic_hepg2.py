import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_dic_hepg2_loader


sys.path.append("..")


def check_dic_hepg2():
    from util import ROOT

    loader = get_dic_hepg2_loader(
        path=os.path.join(ROOT, "dic_hepg2"),
        split="test",
        patch_shape=(512, 512),
        batch_size=2,
        download=True,
    )
    check_loader(loader, 8)


if __name__ == "__main__":
    check_dic_hepg2()
