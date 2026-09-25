import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical.ctrus import get_ctrus_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_ctrus():
    loader = get_ctrus_loader(
        path=os.path.join(DATA_ROOT, "ctrus"),
        patch_shape=(512, 512),
        batch_size=2,
        resize_inputs=False,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_ctrus()
