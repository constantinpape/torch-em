import os

from torch_em.data import MinInstanceSampler
from torch_em.util.debug import check_loader
from torch_em.data.datasets.medical import get_rcc_aid_loader


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def check_rcc_aid():
    loader = get_rcc_aid_loader(
        path=os.path.join(DATA_ROOT, "rcc_aid"),
        patch_shape=(32, 512, 512),
        batch_size=2,
        download=True,
        sampler=MinInstanceSampler(),
    )

    check_loader(loader, 8, plt=True, save_path="./test.png")


if __name__ == "__main__":
    check_rcc_aid()
