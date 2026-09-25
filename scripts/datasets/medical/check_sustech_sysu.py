import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_sustech_sysu_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_sustech_sysu():
    loader = get_sustech_sysu_loader(
        path=os.path.join(DATA_ROOT, "sustech_sysu"),
        batch_size=1,
        patch_shape=(512, 512),
        ndim=2,
        download=True,
    )
    check_loader(loader, 8, plt=True, save_path="./test_sustech_sysu.png")


if __name__ == "__main__":
    check_sustech_sysu()
