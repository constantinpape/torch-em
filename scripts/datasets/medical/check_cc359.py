import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.cc359 import get_cc359_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_cc359():
    loader = get_cc359_loader(
        path=os.path.join(DATA_ROOT, "cc359"),
        patch_shape=(1, 256, 256),
        batch_size=1,
        sampler=MinInstanceSampler(),
        ndim=2,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_cc359()
