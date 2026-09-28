import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_multiphoton_liver_loader


sys.path.append("..")


def check_multiphoton_liver():
    from util import ROOT

    for target in ["bc", "sinusoids", "nuclei", "sinusoid_fill"]:
        loader = get_multiphoton_liver_loader(
            path=os.path.join(ROOT, "multiphoton_liver"),
            batch_size=2,
            patch_shape=(32, 256, 256),
            target=target,
            samples=["G2"],
            download=True,
        )
        check_loader(loader, 4, instance_labels=(target == "nuclei"))


if __name__ == "__main__":
    check_multiphoton_liver()
