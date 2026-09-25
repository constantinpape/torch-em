import os

from torch_em.data import MinForegroundSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_pfus1_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_pfus1():
    loader = get_pfus1_loader(
        path=os.path.join(DATA_ROOT, "pfus1"),
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=True,
        download=True,
        sampler=MinForegroundSampler(min_fraction=0.001),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_pfus1()
