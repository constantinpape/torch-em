import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.histopathology.ecm_phenotyping import get_ecm_phenotyping_loader


sys.path.append("..")


def check_ecm_phenotyping():
    from util import ROOT

    loader = get_ecm_phenotyping_loader(
        path=os.path.join(ROOT, "ecm_phenotyping"),
        patch_shape=(256, 256),
        batch_size=1,
        slides=["slide1"],
        download=True,
    )

    check_loader(loader, 8, instance_labels=True, rgb=False, plt=True, save_path="check_ecm_phenotyping.png")


if __name__ == "__main__":
    check_ecm_phenotyping()
