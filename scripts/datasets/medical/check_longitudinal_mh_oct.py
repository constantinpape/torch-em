import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_longitudinal_mh_oct_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_longitudinal_mh_oct():
    loader = get_longitudinal_mh_oct_loader(
        path=os.path.join(DATA_ROOT, "longitudinal_mh_oct"),
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./longitudinal_mh_oct.png")


if __name__ == "__main__":
    check_longitudinal_mh_oct()
