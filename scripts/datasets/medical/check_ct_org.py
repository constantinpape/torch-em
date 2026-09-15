import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.ct_org import get_ct_org_loader


sys.path.append("..")


def check_ct_org():
    from util import ROOT

    loader = get_ct_org_loader(
        path=os.path.join(ROOT, "ct_org"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        split="test",
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_ct_org()
