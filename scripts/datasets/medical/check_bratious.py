import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_bratious_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_bratious():
    loader = get_bratious_loader(
        path=os.path.join(DATA_ROOT, "bratious"),
        batch_size=2,
        patch_shape=(512, 512),
        resize_inputs=True,
        download=True,
    )
    check_loader(loader, 8, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_bratious()
