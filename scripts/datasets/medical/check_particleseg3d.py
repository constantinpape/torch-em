import os
import sys

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.particleseg3d import get_particleseg3d_loader


sys.path.append("..")


def check_particleseg3d():
    from util import ROOT

    loader = get_particleseg3d_loader(
        path=os.path.join(ROOT, "particleseg3d"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        split="test",
        ndim=2,
        download=True,
        sampler=MinInstanceSampler(min_num_instances=3),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_particleseg3d()
