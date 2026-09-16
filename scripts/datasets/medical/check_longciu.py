import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.longciu import get_longciu_loader


sys.path.append("..")


def check_longciu():
    from util import ROOT

    loader = get_longciu_loader(
        path=os.path.join(ROOT, "longciu"),
        patch_shape=(1, 512, 512),
        batch_size=2,
        split="train",
        annotator="staple",
        ndim=2,
        resize_inputs=False,
        download=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_longciu()
