import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_cyst_x_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_cyst_x():
    loader = get_cyst_x_loader(
        path=os.path.join(DATA_ROOT, "cyst_x"),
        batch_size=1,
        patch_shape=(32, 128, 128),
        sequence="t1",
        n_cases=4,
        download=True,
    )
    check_loader(loader, 4, instance_labels=True, plt=True, save_path="test.png")


if __name__ == "__main__":
    check_cyst_x()
