import os
import sys

from torch_em.data import MinForegroundSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.atlas_stroke import get_atlas_stroke_loader


sys.path.append("..")


def check_atlas_stroke():
    from util import ROOT

    loader = get_atlas_stroke_loader(
        path=os.path.join(ROOT, "atlas_stroke"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        ndim=2,
        resize_inputs=True,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.001, p_reject=0.95),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_atlas_stroke()
