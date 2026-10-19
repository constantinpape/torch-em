import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_lipid_droplet_histology_loader


sys.path.append("..")


def check_lipid_droplet_histology():
    from util import ROOT

    loader = get_lipid_droplet_histology_loader(
        path=os.path.join(ROOT, "lipid_droplet_histology"),
        patch_shape=(256, 256),
        batch_size=2,
        download=True,
    )

    check_loader(loader, 8, instance_labels=True)


if __name__ == "__main__":
    check_lipid_droplet_histology()
