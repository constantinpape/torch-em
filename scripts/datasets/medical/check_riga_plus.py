import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_riga_plus_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_riga_plus():
    loader = get_riga_plus_loader(
        path=os.path.join(DATA_ROOT, "riga_plus"),
        patch_shape=(512, 512),
        batch_size=2,
        split="train",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, rgb=True, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_riga_plus()
