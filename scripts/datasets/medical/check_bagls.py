import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.bagls import get_bagls_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_bagls():
    loader = get_bagls_loader(
        path=os.path.join(DATA_ROOT, "bagls"),
        patch_shape=(256, 256),
        batch_size=2,
        split="test",
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_bagls()
