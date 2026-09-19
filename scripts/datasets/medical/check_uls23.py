import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.uls23 import get_uls23_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_uls23():
    loader = get_uls23_loader(
        path=os.path.join(DATA_ROOT, "uls23"),
        batch_size=1,
        patch_shape=(32, 128, 128),
        source="kits21",
        ndim=3,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_uls23.png")


if __name__ == "__main__":
    check_uls23()
