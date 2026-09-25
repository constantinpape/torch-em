import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_muscle_us_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_muscle_us():
    loader = get_muscle_us_loader(
        path=os.path.join(DATA_ROOT, "muscle_us"),
        patch_shape=(512, 512),
        batch_size=2,
        muscle=None,
        category=None,
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_muscle_us()
