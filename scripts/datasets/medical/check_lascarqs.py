import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.lascarqs import get_lascarqs_loader


sys.path.append("..")


def check_lascarqs():
    from util import ROOT

    loader = get_lascarqs_loader(
        path=os.path.join(ROOT, "lascarqs"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        task="task1",
        sampler=MinInstanceSampler(),
        ndim=2,
        download=False,
    )

    check_loader(loader, 8, plt=True, save_path="./lascarqs.png")


if __name__ == "__main__":
    check_lascarqs()
