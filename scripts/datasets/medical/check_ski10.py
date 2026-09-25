import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.ski10 import get_ski10_loader


sys.path.append("..")


def check_ski10():
    from util import ROOT

    loader = get_ski10_loader(
        path=os.path.join(ROOT, "ski10"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        resize_inputs=True,
        download=True,
        sampler=MinInstanceSampler(min_num_instances=5),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_ski10()
