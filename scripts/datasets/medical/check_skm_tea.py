import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.skm_tea import get_skm_tea_loader


sys.path.append("..")


def check_skm_tea():
    from util import ROOT

    loader = get_skm_tea_loader(
        path=os.path.join(ROOT, "skm_tea"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        split="train",
        echo="echo1",
        ndim=2,
        download=True,
        sampler=MinInstanceSampler(min_num_instances=3),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_skm_tea()
