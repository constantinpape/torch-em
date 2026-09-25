import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_rossa_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_rossa():
    loader = get_rossa_loader(
        path=os.path.join(DATA_ROOT, "rossa"),
        patch_shape=(320, 320),
        batch_size=2,
        annotation="manual",
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./rossa.png")


if __name__ == "__main__":
    check_rossa()
