import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.busbra import get_busbra_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_busbra():
    loader = get_busbra_loader(
        path=os.path.join(DATA_ROOT, "busbra"),
        patch_shape=(256, 256),
        batch_size=1,
        sampler=MinInstanceSampler(),
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_busbra()
