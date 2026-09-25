import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_woives_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_woives():
    loader = get_woives_loader(
        path=os.path.join(DATA_ROOT, "woives"),
        batch_size=2,
        patch_shape=(512, 512),
        split="train",
        fold=0,
        download=True,
    )
    check_loader(loader, 4, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_woives()
