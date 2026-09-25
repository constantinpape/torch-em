import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.plaque_us import get_plaque_us_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_plaque_us():
    loader = get_plaque_us_loader(
        path=os.path.join(DATA_ROOT, "plaque_us"),
        batch_size=2,
        patch_shape=(512, 512),
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_plaque_us.png")


if __name__ == "__main__":
    check_plaque_us()
