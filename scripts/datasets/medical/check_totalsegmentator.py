import os
import sys

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.totalsegmentator import get_totalsegmentator_loader


sys.path.append("..")


def check_totalsegmentator():
    from util import ROOT

    loader = get_totalsegmentator_loader(
        path=os.path.join(ROOT, "totalsegmentator"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        split="train",
        ndim=2,
        download=True,
        sampler=MinInstanceSampler(min_num_instances=3),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_totalsegmentator()
