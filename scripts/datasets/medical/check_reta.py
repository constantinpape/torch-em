import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.reta import get_reta_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_reta():
    loader = get_reta_loader(
        path=os.path.join(DATA_ROOT, "reta"),
        batch_size=2,
        patch_shape=(512, 512),
        split="train",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_reta.png")


if __name__ == "__main__":
    check_reta()
