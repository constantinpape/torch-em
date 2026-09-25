import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_pyropia_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_pyropia():
    loader = get_pyropia_loader(
        path=os.path.join(DATA_ROOT, "pyropia"),
        batch_size=2,
        patch_shape=(512, 512),
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_pyropia()
