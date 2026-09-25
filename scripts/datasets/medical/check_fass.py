import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.fass import get_fass_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_fass():
    loader = get_fass_loader(
        path=os.path.join(DATA_ROOT, "fass"),
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_fass()
