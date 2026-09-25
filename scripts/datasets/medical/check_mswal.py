import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical.mswal import get_mswal_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_mswal():
    loader = get_mswal_loader(
        path=os.path.join(DATA_ROOT, "mswal"),
        batch_size=1,
        patch_shape=(32, 256, 256),
        ndim=3,
        sampler=MinInstanceSampler(),
        download=False,
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_mswal()
