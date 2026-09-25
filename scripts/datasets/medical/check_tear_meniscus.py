import os

from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_tear_meniscus_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_tear_meniscus():
    loader = get_tear_meniscus_loader(
        path=os.path.join(DATA_ROOT, "tear_meniscus"),
        patch_shape=(256, 256),
        batch_size=2,
        ndim=2,
        resize_inputs=True,
        download=True,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_tear_meniscus()
