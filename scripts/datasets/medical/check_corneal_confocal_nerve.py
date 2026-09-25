import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_corneal_confocal_nerve_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_corneal_confocal_nerve():
    loader = get_corneal_confocal_nerve_loader(
        path=os.path.join(DATA_ROOT, "corneal_confocal_nerve"),
        patch_shape=(256, 256),
        batch_size=2,
        ndim=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_corneal_confocal_nerve()
