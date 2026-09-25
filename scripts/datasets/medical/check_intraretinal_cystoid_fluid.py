import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.intraretinal_cystoid_fluid import get_intraretinal_cystoid_fluid_loader


sys.path.append("..")


def check_intraretinal_cystoid_fluid():
    from util import ROOT

    loader = get_intraretinal_cystoid_fluid_loader(
        path=os.path.join(ROOT, "intraretinal_cystoid_fluid"),
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, rgb=True, plt=True, save_path="intraretinal_cystoid_fluid_check.png")


if __name__ == "__main__":
    check_intraretinal_cystoid_fluid()
