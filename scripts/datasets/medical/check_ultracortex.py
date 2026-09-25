import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_ultracortex_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_ultracortex():
    loader = get_ultracortex_loader(
        path=os.path.join(DATA_ROOT, "ultracortex"),
        patch_shape=(1, 256, 256),
        batch_size=2,
        ndim=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_ultracortex()
