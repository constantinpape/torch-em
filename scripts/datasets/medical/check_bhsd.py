import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_bhsd_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_bhsd():
    loader = get_bhsd_loader(
        path=os.path.join(DATA_ROOT, "bhsd"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./bhsd.png")


if __name__ == "__main__":
    check_bhsd()
