import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.mindboggle101 import get_mindboggle101_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_mindboggle101():
    loader = get_mindboggle101_loader(
        path=os.path.join(DATA_ROOT, "mindboggle101"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        sampler=MinInstanceSampler(),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_mindboggle101()
