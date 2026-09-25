import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.atlas_liver import get_atlas_liver_loader


sys.path.append("..")


def check_atlas_liver():
    from util import ROOT

    loader = get_atlas_liver_loader(
        path=os.path.join(ROOT, "atlas_liver"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_atlas_liver()
