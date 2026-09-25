import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_lumbar_spine_us_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_lumbar_spine_us():
    loader = get_lumbar_spine_us_loader(
        path=os.path.join(DATA_ROOT, "lumbar_spine_us"),
        batch_size=1,
        patch_shape=(512, 512),
        probe="all",
        ndim=2,
        download=True,
    )
    check_loader(loader, 8, plt=True, save_path="./test_lumbar_spine_us.png")


if __name__ == "__main__":
    check_lumbar_spine_us()
