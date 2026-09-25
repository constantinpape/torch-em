import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_stare_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_stare():
    loader = get_stare_loader(
        path=os.path.join(DATA_ROOT, "stare"),
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, rgb=True, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_stare()
