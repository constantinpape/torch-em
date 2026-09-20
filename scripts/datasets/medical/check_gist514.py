import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.gist514 import get_gist514_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_gist514():
    loader = get_gist514_loader(
        path=os.path.join(DATA_ROOT, "gist514"),
        patch_shape=(256, 256),
        batch_size=1,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test_gist514.png")


if __name__ == "__main__":
    check_gist514()
