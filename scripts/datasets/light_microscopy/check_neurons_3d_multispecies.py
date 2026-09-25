import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_neurons_3d_multispecies_loader


sys.path.append("..")


def check_neurons_3d_multispecies():
    from util import ROOT

    loader = get_neurons_3d_multispecies_loader(
        path=os.path.join(ROOT, "neurons_3d_multispecies"),
        batch_size=1,
        patch_shape=(16, 256, 256),
        download=True,
    )

    check_loader(loader, 4, instance_labels=True, plt=True, save_path="neurons_3d_multispecies.png")


if __name__ == "__main__":
    check_neurons_3d_multispecies()
