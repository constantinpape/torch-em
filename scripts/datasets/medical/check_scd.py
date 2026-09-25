import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_scd_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_scd():
    loader = get_scd_loader(
        path=os.path.join(DATA_ROOT, "scd"),
        patch_shape=(256, 256),
        batch_size=1,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./scd.png")


if __name__ == "__main__":
    check_scd()
