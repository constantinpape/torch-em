import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.afio import get_afio_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_afio():
    loader = get_afio_loader(
        path=os.path.join(DATA_ROOT, "afio"),
        batch_size=2,
        patch_shape=(512, 512),
        task="vessels",
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_afio.png")


if __name__ == "__main__":
    check_afio()
