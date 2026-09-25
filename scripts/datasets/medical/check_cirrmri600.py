import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_cirrmri600_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_cirrmri600():
    loader = get_cirrmri600_loader(
        path=os.path.join(DATA_ROOT, "cirrmri600"),
        batch_size=2,
        patch_shape=(16, 256, 256),
        split="train",
        sequence="T2",
        download=True,
    )
    check_loader(loader, 4, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_cirrmri600()
