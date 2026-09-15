import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.mediastinal_ct import get_mediastinal_ct_loader


sys.path.append("..")


def check_mediastinal_ct():
    from util import ROOT

    for task in ["lymph_nodes", "structures"]:
        loader = get_mediastinal_ct_loader(
            path=os.path.join(ROOT, "mediastinal_ct"),
            patch_shape=(1, 512, 512),
            batch_size=1,
            task=task,
            ndim=2,
            download=True,
        )
        check_loader(loader, 8, plt=True, save_path=f"./test_{task}.png")


if __name__ == "__main__":
    check_mediastinal_ct()
