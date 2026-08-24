import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_ecoli_microcolony_lineage_loader


sys.path.append("..")


def check_ecoli_microcolony_lineage():
    # from util import ROOT
    ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

    loader = get_ecoli_microcolony_lineage_loader(
        path=os.path.join(ROOT, "ecoli_microcolony_lineage"),
        batch_size=2,
        genes=["cib"],
        patch_shape=(512, 512),
        download=True,
        sampler=MinInstanceSampler(),
    )
    check_loader(loader, 8, plt=True, instance_labels=True, save_path="./ecoli_microcolony_lineage.png")


if __name__ == "__main__":
    check_ecoli_microcolony_lineage()
