import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.echonet_pediatric import get_echonet_pediatric_loader


sys.path.append("..")


def check_echonet_pediatric():
    from util import ROOT

    loader = get_echonet_pediatric_loader(
        path=os.path.join(ROOT, "echonet_pediatric"),
        patch_shape=(112, 112),
        batch_size=2,
        view="A4C",
        ndim=2,
        download=False,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_echonet_pediatric()
