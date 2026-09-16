import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.trackrad import get_trackrad_loader


sys.path.append("..")


def check_trackrad():
    from util import ROOT

    loader = get_trackrad_loader(
        path=os.path.join(ROOT, "trackrad"),
        patch_shape=(256, 256, 1),
        batch_size=2,
        split="training",
        resize_inputs=False,
        download=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8, plt=True, save_path="./trackrad.png")


if __name__ == "__main__":
    check_trackrad()
