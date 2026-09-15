import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.upenn_gbm import get_upenn_gbm_loader


sys.path.append("..")


def check_upenn_gbm():
    from util import ROOT

    loader = get_upenn_gbm_loader(
        path=os.path.join(ROOT, "upenn_gbm"),
        patch_shape=(1, 240, 240),
        batch_size=1,
        ndim=2,
        modality="T1GD",
        segmentation="manual",
        sampler=MinInstanceSampler(min_num_instances=3),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_upenn_gbm()
