import os
import sys

from torch_em.data.datasets import get_tumor_spheroid_loader
from torch_em.util.debug import check_loader

sys.path.append("..")


def check_tumor_spheroid():
    from util import ROOT

    path = os.path.join(ROOT, "tumor_spheroid_em")

    loader = get_tumor_spheroid_loader(
        path, patch_shape=(512, 512), batch_size=1, source="2d_manual",
        resolution="50-50-50", target="cells", download=True,
    )
    check_loader(loader, 8, instance_labels=True, plt=True, save_path="./check_tumor_spheroid_em_cells.png")

    loader_nuclei = get_tumor_spheroid_loader(
        path, patch_shape=(512, 512), batch_size=1, source="2d_manual",
        resolution="50-50-50", target="nuclei", download=True,
    )
    check_loader(loader_nuclei, 8, instance_labels=True, plt=True, save_path="./check_tumor_spheroid_em_nuclei.png")


if __name__ == "__main__":
    check_tumor_spheroid()
