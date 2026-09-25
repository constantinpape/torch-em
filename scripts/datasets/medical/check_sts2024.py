import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.sts2024 import get_sts2024_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_sts2024():
    loader = get_sts2024_loader(
        path=os.path.join(DATA_ROOT, "sts2024"),
        patch_shape=(512, 512),
        batch_size=1,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_sts2024.png")


if __name__ == "__main__":
    check_sts2024()
