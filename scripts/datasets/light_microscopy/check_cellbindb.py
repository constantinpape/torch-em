import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_cellbindb_loader


sys.path.append("..")


def check_cellbindb():
    from util import ROOT

    loader = get_cellbindb_loader(
        path=os.path.join(ROOT, "cellbindb"),
        patch_shape=(512, 512),
        batch_size=2,
        data_choice=None,
        download=False,
        shuffle=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_cellbindb()
