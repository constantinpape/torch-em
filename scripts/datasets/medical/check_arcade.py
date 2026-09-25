import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_arcade_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_arcade():
    loader = get_arcade_loader(
        path=os.path.join(DATA_ROOT, "arcade"),
        patch_shape=(512, 512),
        batch_size=1,
        task="syntax",
        split="train",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./arcade.png")


if __name__ == "__main__":
    check_arcade()
